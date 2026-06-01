/*
 * Copyright Cedar Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! A persistent, per-worker-thread cvc5 process driven directly from Rust.
//!
//! The analyzer issues thousands of small, independent SMT queries. The Lean
//! backend spawns a *fresh* cvc5 for every one (`Solver.cvc5` in
//! `Cedar/SymCC/Solver.lean`), and for these cheap queries the process spawn
//! dominates wall time. Instead, we obtain each query's self-contained SMT-LIB
//! script from Lean via the `smtlib_of_check_*` FFI -- which runs the encoder
//! but does *not* spawn a solver -- and feed it to one long-lived cvc5 per rayon
//! worker thread.
//!
//! This is sound because every script produced by the encoder begins with
//! `(reset)`, which returns cvc5 to its initial state. A single process can
//! therefore serve an unbounded sequence of queries with no cross-query state
//! leak -- precisely the clean slate that a per-query spawn used to provide.
//! The decision maps the same way the Lean backend maps it: each analyzer check
//! is `checkUnsatAsserts (verify… …)`, so the property holds **iff the script is
//! UNSAT** (see `Cedar/SymCCOpt.lean` and `Cedar/SymCC/SatUnsat.lean`).

use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

/// How long `Drop` waits for cvc5 to exit cleanly after `(exit)` before it
/// force-kills the child, so a stuck or desynced solver can never hang teardown.
const SHUTDOWN_GRACE: Duration = Duration::from_millis(200);
/// Poll interval while waiting for the clean exit above.
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(5);
/// Defensive cap on consecutive blank output lines before the stream is treated
/// as desynced. A well-behaved `--quiet` cvc5 emits none here.
const MAX_BLANK_LINES: u32 = 1024;

/// A solver failure encountered while reusing a cvc5 process.
#[derive(Debug)]
pub enum SolverError {
    /// cvc5 returned `unknown` (mirrors the Lean backend, which errors here).
    /// Carries cvc5's `:reason-unknown` when available, for diagnostics.
    Unknown(String),
    /// The solver process could not be spawned, or its pipe failed / desynced.
    Io(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Unknown(reason) => write!(f, "cvc5 returned unknown: {reason}"),
            SolverError::Io(msg) => write!(f, "cvc5 process error: {msg}"),
        }
    }
}

impl std::error::Error for SolverError {}

enum Decision {
    Sat,
    Unsat,
    Unknown(String),
}

/// A live cvc5 child process with its stdin/stdout pipes.
struct Cvc5 {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Cvc5 {
    /// Spawn cvc5 with the same arguments the Lean backend uses
    /// (`Solver.cvc5`): quiet mode, SMT-LIB input. The executable is read from
    /// the `CVC5` environment variable, matching the Lean backend.
    fn spawn() -> Result<Self, SolverError> {
        let path = std::env::var("CVC5")
            .map_err(|_| SolverError::Io("CVC5 environment variable not defined.".to_string()))?;
        let mut child = Command::new(path)
            .args(["--quiet", "--lang", "smt"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Surface cvc5 diagnostics (version/parse/feature errors) on stderr
            // rather than discarding them; `--quiet` keeps this silent otherwise.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| SolverError::Io(format!("failed to spawn cvc5: {e}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| SolverError::Io("cvc5 stdin pipe missing".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SolverError::Io("cvc5 stdout pipe missing".to_string()))?;
        Ok(Cvc5 {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// Send one self-contained SMT-LIB script (which begins with `(reset)` and
    /// ends with a single `(check-sat)`) and read back the decision.
    fn solve(&mut self, script: &str) -> Result<Decision, SolverError> {
        let io = |e: std::io::Error| SolverError::Io(e.to_string());
        self.stdin.write_all(script.as_bytes()).map_err(io)?;
        if !script.ends_with('\n') {
            self.stdin.write_all(b"\n").map_err(io)?;
        }
        self.stdin.flush().map_err(io)?;

        // `--quiet` suppresses `success` acknowledgements, so the next non-empty
        // output line is the `(check-sat)` result. The script emits exactly one
        // `(check-sat)` and no `(get-model)`, so exactly one decision is read.
        let mut line = String::new();
        let mut blank_lines = 0u32;
        loop {
            line.clear();
            if self.stdout.read_line(&mut line).map_err(io)? == 0 {
                return Err(SolverError::Io(
                    "cvc5 closed its output unexpectedly".to_string(),
                ));
            }
            match line.trim() {
                "" => {
                    // A misbehaving solver emitting endless blank lines must not
                    // wedge this worker in the loop forever.
                    blank_lines += 1;
                    if blank_lines > MAX_BLANK_LINES {
                        return Err(SolverError::Io(
                            "cvc5 produced only blank output (protocol desync)".to_string(),
                        ));
                    }
                    continue;
                }
                "unsat" => return Ok(Decision::Unsat),
                "sat" => return Ok(Decision::Sat),
                "unknown" => return Ok(Decision::Unknown(self.reason_unknown())),
                other => {
                    return Err(SolverError::Io(format!(
                        "unrecognized cvc5 output: {other:?}"
                    )));
                }
            }
        }
    }

    /// Best-effort `(get-info :reason-unknown)` so an `unknown` decision carries
    /// cvc5's own explanation. Called only on the cold `unknown` path, after
    /// which the process is discarded, so a failed read here is harmless.
    fn reason_unknown(&mut self) -> String {
        const FALLBACK: &str = "reason unavailable";
        if self
            .stdin
            .write_all(b"(get-info :reason-unknown)\n")
            .and_then(|()| self.stdin.flush())
            .is_err()
        {
            return FALLBACK.to_string();
        }
        let mut line = String::new();
        match self.stdout.read_line(&mut line) {
            Ok(n) if n > 0 && !line.trim().is_empty() => line.trim().to_string(),
            _ => FALLBACK.to_string(),
        }
    }
}

impl Drop for Cvc5 {
    fn drop(&mut self) {
        // Best-effort clean shutdown: ask cvc5 to exit, then reap the child so it
        // never lingers.
        let _ = self.stdin.write_all(b"(exit)\n");
        let _ = self.stdin.flush();

        // Never block teardown on a stuck or desynced solver: poll for a clean
        // exit for a short grace period, then force-kill and reap.
        let deadline = Instant::now() + SHUTDOWN_GRACE;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return, // exited on its own
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(SHUTDOWN_POLL_INTERVAL);
                }
                // Grace elapsed, or its state is unknowable: stop waiting.
                _ => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

thread_local! {
    /// One reusable cvc5 process per OS (rayon worker) thread. Created lazily on
    /// first use and reused for every subsequent query on that thread. Confining
    /// it to a thread keeps the single process's stdin/stdout free of
    /// interleaving without any locking.
    static SOLVER: RefCell<Option<Cvc5>> = const { RefCell::new(None) };
}

/// Decide a `checkUnsat`-style verification by reusing this worker thread's cvc5
/// process. `script` is the output of a `smtlib_of_check_*` FFI call.
///
/// Returns:
/// * `Ok(Some(true))`  -- the script is UNSAT, i.e. the checked property holds;
/// * `Ok(Some(false))` -- the script is SAT, i.e. the property does not hold;
/// * `Ok(None)`        -- the query was *trivially decided* by the encoder, which
///   emits an empty script in that case. No solver is run; the caller decides it
///   directly from the assertions (see `Analyzer::check_reusing_solver`).
///
/// On any solver failure the thread-local process is discarded, so a desynced or
/// closed pipe can never poison subsequent queries on this thread -- the next
/// call simply starts a fresh process.
pub fn check_unsat(script: &str) -> Result<Option<bool>, SolverError> {
    if script.trim().is_empty() {
        return Ok(None);
    }
    SOLVER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Cvc5::spawn()?);
        }
        let decision = slot
            .as_mut()
            .expect("solver was just ensured to be present")
            .solve(script);
        match decision {
            Ok(Decision::Unsat) => Ok(Some(true)),
            Ok(Decision::Sat) => Ok(Some(false)),
            Ok(Decision::Unknown(reason)) => {
                *slot = None;
                Err(SolverError::Unknown(reason))
            }
            Err(e) => {
                *slot = None;
                Err(e)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests drive a real cvc5 process, so they require the `CVC5` env var
    /// (the same requirement as the analyzer's integration tests). When it is
    /// unset we skip rather than fail, so solver-free `--lib` runs still pass.
    fn cvc5_available() -> bool {
        if std::env::var("CVC5").is_ok() {
            true
        } else {
            eprintln!("skipping reused_solver test: CVC5 environment variable not set");
            false
        }
    }

    #[test]
    fn empty_script_is_trivially_decided_without_a_process() {
        // The encoder emits an empty script for syntactically-decided queries;
        // `check_unsat` reports that as `None` without spawning a solver, so this
        // case is exercised even where no cvc5 is available.
        assert!(matches!(check_unsat("  \n\t "), Ok(None)));
    }

    #[test]
    fn parses_unsat_sat_and_reuses_the_process_across_queries() {
        if !cvc5_available() {
            return;
        }
        // All queries run on this one test thread, so they share a single reused
        // cvc5 process; each script is self-contained and begins with `(reset)`,
        // exactly as the encoder emits.

        // `(assert false)` is UNSAT => the checked property holds.
        assert_eq!(
            check_unsat("(reset)\n(assert false)\n(check-sat)\n").expect("first query"),
            Some(true),
        );

        // A satisfiable assertion is SAT => the property does not hold. Reusing
        // the same process also proves a second non-empty query works.
        assert_eq!(
            check_unsat("(reset)\n(declare-fun x () Bool)\n(assert x)\n(check-sat)\n")
                .expect("second query"),
            Some(false),
        );

        // `(reset)` must have cleared the previous `declare-fun`: this query
        // re-declares nothing and must still be answered by the same process,
        // proving reset-based reuse with no cross-query state leak.
        assert_eq!(
            check_unsat("(reset)\n(assert (not true))\n(check-sat)\n").expect("third query"),
            Some(true),
        );
    }
}
