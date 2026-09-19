//! The error type, whose variants name the kind of failure rather than the
//! place it happened, so the Python bindings can map each to the exception a
//! caller would expect.

use core::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// `RangeError`
    Range(String),
    /// A division that cannot be carried out exactly.
    Division(String),
    /// `TypeError`
    Type(String),
    /// `ParseError`
    Parse(String),
    /// A puzzle cannot be described or turned as asked. Distinct from the
    /// arithmetic failures above: it is raised by things that are about the
    /// puzzle rather than the numbers, such as reading a state as a sticker
    /// array.
    State(String),
    /// Plain `Error`
    Other(String),
}

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Error::Range(m)
            | Error::Division(m)
            | Error::Type(m)
            | Error::Parse(m)
            | Error::State(m)
            | Error::Other(m) => m,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Error::Range(_) => "RangeError",
            Error::Division(_) => "DivisionError",
            Error::Type(_) => "TypeError",
            Error::Parse(_) => "ParseError",
            Error::State(_) => "PuzzleError",
            Error::Other(_) => "Error",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

/// Logs but never panics.
///
/// Several algorithms here deliberately continue past a false assertion (such as
/// dropping a degenerate face without halting), so turning these into panics
/// would change control flow rather than just noise.
#[macro_export]
macro_rules! console_assert {
    ($cond:expr, $($arg:tt)*) => {
        if cfg!(debug_assertions) && !$cond {
            eprintln!("Assertion failed: {}", format_args!($($arg)*));
        }
    };
    ($cond:expr) => {
        if cfg!(debug_assertions) && !$cond {
            eprintln!("Assertion failed: console.assert");
        }
    };
}
