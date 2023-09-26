use std::fmt;

pub mod hugr_to_mlir;
pub mod mlir;

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    MeliorError(melior::Error),
    StringError(String),
}

impl From<melior::Error> for Error {
    fn from(e: melior::Error) -> Self {
        Self::MeliorError(e)
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Self::StringError(e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::MeliorError(e) => e.fmt(formatter),
            Self::StringError(e) => write!(formatter, "{}", e),
        }
    }
}

impl std::error::Error for Error {}
