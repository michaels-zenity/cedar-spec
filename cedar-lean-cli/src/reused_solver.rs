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

/// A solver failure encountered while reusing a cvc5 process.
#[derive(Debug)]
pub enum SolverError {
    /// cvc5 returned `unknown` (mirrors the Lean backend, which errors here).
    Unknown,
    /// The solver process could not be spawned, or its pipe failed / desynced.
    Io(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Unknown => write!(f, "cvc5 returned unknown"),
            SolverError::Io(msg) => write!(f, "cvc5 process error: {msg}"),
        }
    }
}

impl std::error::Error for SolverError {}

enum Decision {
    Sat,
    Unsat,
    Unknown,
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
            .stderr(Stdio::null())
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
        loop {
            line.clear();
            if self.stdout.read_line(&mut line).map_err(io)? == 0 {
                return Err(SolverError::Io(
                    "cvc5 closed its output unexpectedly".to_string(),
                ));
            }
            match line.trim() {
                "" => continue,
                "unsat" => return Ok(Decision::Unsat),
                "sat" => return Ok(Decision::Sat),
                "unknown" => return Ok(Decision::Unknown),
                other => {
                    return Err(SolverError::Io(format!(
                        "unrecognized cvc5 output: {other:?}"
                    )));
                }
            }
        }
    }
}

impl Drop for Cvc5 {
    fn drop(&mut self) {
        // Best-effort clean shutdown, then reap the child so it never lingers.
        let _ = self.stdin.write_all(b"(exit)\n");
        let _ = self.stdin.flush();
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
///   emits an empty script in that case. The caller must fall back to the
///   authoritative `run_check_*` FFI, which returns the right answer without
///   needing a solver decision. (This path is rare and still spawns one cvc5, so
///   it is no worse than the baseline for those queries.)
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
            Ok(Decision::Unknown) => {
                *slot = None;
                Err(SolverError::Unknown)
            }
            Err(e) => {
                *slot = None;
                Err(e)
            }
        }
    })
}
