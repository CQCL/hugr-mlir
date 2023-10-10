#![allow(unused_imports)]
use std::fmt;

pub mod hugr_to_mlir;
pub mod mlir;
pub mod translate;

// #[derive(Debug)]
// pub enum Error {
//     MeliorError(melior::Error),
//     StringError(String),
//     SerdeJsonError(serde_json::Error),
//     SerdeRmpError(rmp_serde::decode::Error),
// }

// impl From<melior::Error> for Error {
//     fn from(e: melior::Error) -> Self {
//         Self::MeliorError(e)
//     }
// }

// impl From<serde_json::Error> for Error {
//     fn from(e: serde_json::Error) -> Self {
//         Self::SerdeJsonError(e)
//     }
// }

// impl From<rmp_serde::decode::Error> for Error {
//     fn from(e: rmp_serde::decode::Error) -> Self {
//         Self::SerdeRmpError(e)
//     }
// }

// impl From<String> for Error {
//     fn from(e: String) -> Self {
//         Self::StringError(e)
//     }
// }

// impl fmt::Display for Error {
//     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
//         match self {
//             Self::MeliorError(e) => e.fmt(formatter),
//             Self::StringError(e) => write!(formatter, "{}", e),
//             Self::SerdeJsonError(e) => write!(formatter, "{}", e),
//             Self::SerdeRmpError(e) => write!(formatter, "{}", e),
//         }
//     }
// }

// impl std::error::Error for Error {}

// type Result<X, E = Error> = core::result::Result<X,E>;

pub use anyhow::{Error, Result};
