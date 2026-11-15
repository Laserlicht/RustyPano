// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Where a panorama or tour file comes from: a local path, a file handed over
//! in memory (browser file picker / drag and drop) or a web address.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type Bytes = Arc<[u8]>;

#[derive(Clone)]
// `Memory` and `Url` are created by the browser build only.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub enum Source {
    Path(PathBuf),
    /// A file passed in memory, plus the files opened together with it, so
    /// that a tour can find its images by name.
    Memory { name: String, data: Bytes, siblings: Arc<HashMap<String, Bytes>> },
    Url(String),
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Source::Path(a), Source::Path(b)) => a == b,
            (Source::Url(a), Source::Url(b)) => a == b,
            (Source::Memory { name: a, data: da, .. }, Source::Memory { name: b, data: db, .. }) => {
                a == b && Arc::ptr_eq(da, db)
            }
            _ => false,
        }
    }
}

fn last_component(s: &str) -> &str {
    let s = s.split(['?', '#']).next().unwrap_or(s);
    s.rsplit(['/', '\\']).next().unwrap_or(s)
}

impl Source {
    /// A group of files opened together; returns one source per file, all
    /// sharing the same sibling table.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub fn memory_group(files: Vec<(String, Vec<u8>)>) -> Vec<Source> {
        let files: Vec<(String, Bytes)> = files.into_iter().map(|(n, d)| (n, Bytes::from(d))).collect();
        let siblings: Arc<HashMap<String, Bytes>> =
            Arc::new(files.iter().map(|(n, d)| (n.to_lowercase(), d.clone())).collect());
        files
            .into_iter()
            .map(|(name, data)| Source::Memory { name, data, siblings: siblings.clone() })
            .collect()
    }

    /// File name for titles and status text.
    pub fn file_name(&self) -> String {
        match self {
            Source::Path(p) => p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            Source::Memory { name, .. } => name.clone(),
            Source::Url(u) => last_component(u).to_string(),
        }
    }

    /// Full description for error messages.
    pub fn describe(&self) -> String {
        match self {
            Source::Path(p) => p.display().to_string(),
            Source::Memory { name, .. } => name.clone(),
            Source::Url(u) => u.clone(),
        }
    }

    pub fn extension(&self) -> String {
        let name = self.file_name();
        match name.rsplit_once('.') {
            Some((_, ext)) => ext.to_lowercase(),
            None => String::new(),
        }
    }

    pub fn is_fsv(&self) -> bool {
        self.extension() == "fsv"
    }

    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Source::Path(p) => Some(p),
            _ => None,
        }
    }

    /// Resolve a file name written in a tour file relative to this source.
    pub fn resolve(&self, reference: &str) -> Result<Source, String> {
        let reference = reference.trim().replace('\\', "/");
        match self {
            Source::Path(p) => {
                let r = PathBuf::from(&reference);
                Ok(Source::Path(if r.is_absolute() {
                    r
                } else {
                    p.parent().unwrap_or(Path::new(".")).join(r)
                }))
            }
            Source::Url(u) => {
                if reference.contains("://") {
                    return Ok(Source::Url(reference));
                }
                let base = u.split(['?', '#']).next().unwrap_or(u);
                let dir = &base[..base.rfind('/').map_or(0, |i| i + 1)];
                Ok(Source::Url(format!("{dir}{reference}")))
            }
            Source::Memory { siblings, .. } => {
                let name = last_component(&reference);
                match siblings.get(&name.to_lowercase()) {
                    Some(data) => Ok(Source::Memory { name: name.to_string(), data: data.clone(), siblings: siblings.clone() }),
                    None => Err(tr!(
                        "{name} was not opened together with the tour. Open the tour file and its images at the same time.",
                        name = name
                    )),
                }
            }
        }
    }

    pub async fn read(&self) -> Result<Bytes, String> {
        match self {
            Source::Memory { data, .. } => Ok(data.clone()),
            #[cfg(not(target_arch = "wasm32"))]
            Source::Path(p) => {
                if !p.exists() {
                    return Err(tr!("No such file: {path}", path = p.display()));
                }
                std::fs::read(p).map(Bytes::from).map_err(|e| format!("{}: {e}", p.display()))
            }
            #[cfg(target_arch = "wasm32")]
            Source::Path(p) => Err(tr!("No such file: {path}", path = p.display())),
            #[cfg(target_arch = "wasm32")]
            Source::Url(u) => crate::web::fetch(u).await.map(Bytes::from),
            #[cfg(not(target_arch = "wasm32"))]
            Source::Url(_) => Err(tr!("Web addresses can only be opened in the browser version.").into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_references() {
        let p = Source::Path(PathBuf::from("/a/b/tour.fsv"));
        assert_eq!(p.resolve("img\\x.jpg").unwrap(), Source::Path(PathBuf::from("/a/b/img/x.jpg")));
        let u = Source::Url("https://h.example/t/tour.fsv?v=1".into());
        assert_eq!(u.resolve("x.jpg").unwrap(), Source::Url("https://h.example/t/x.jpg".into()));
        let group = Source::memory_group(vec![("Tour.fsv".into(), vec![1]), ("Pano.JPG".into(), vec![2])]);
        assert_eq!(&*group[0].resolve("sub/pano.jpg").unwrap().file_name(), "pano.jpg");
        assert!(group[0].resolve("missing.jpg").is_err());
        assert_eq!(u.file_name(), "tour.fsv");
        assert!(u.is_fsv());
    }
}
