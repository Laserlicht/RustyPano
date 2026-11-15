// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Minimal gettext-style translations.
//!
//! The English text is the lookup key: `tr!("Open…")` returns the text in the
//! current language, `tr!("Loaded {n} files", n = count)` additionally fills
//! named `{placeholders}`. Catalogs are TOML files in `i18n/` mapping English
//! to the translated text, embedded at compile time. Missing entries fall back
//! to English.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

/// The user's choice in Settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageChoice {
    /// Follow the operating system's language.
    #[default]
    Auto,
    En,
    De,
}

/// A language with a catalog (English needs none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Lang {
    En = 0,
    De = 1,
}

impl Lang {
    /// The language's name in that language, for the language picker.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::De => "Deutsch",
        }
    }

    fn from_tag(tag: &str) -> Option<Lang> {
        let primary = tag.split(['-', '_', '.', '@']).next()?.to_ascii_lowercase();
        match primary.as_str() {
            "en" => Some(Lang::En),
            "de" => Some(Lang::De),
            _ => None,
        }
    }
}

impl LanguageChoice {
    pub const ALL: [LanguageChoice; 3] = [LanguageChoice::Auto, LanguageChoice::En, LanguageChoice::De];

    pub fn resolve(self) -> Lang {
        match self {
            LanguageChoice::Auto => detect(),
            LanguageChoice::En => Lang::En,
            LanguageChoice::De => Lang::De,
        }
    }
}

static CURRENT: AtomicU8 = AtomicU8::new(Lang::En as u8);

/// First supported language among the system's preferred languages.
pub fn detect() -> Lang {
    sys_locale::get_locales().find_map(|l| Lang::from_tag(&l)).unwrap_or(Lang::En)
}

pub fn apply(choice: LanguageChoice) {
    CURRENT.store(choice.resolve() as u8, Ordering::Relaxed);
}

pub fn current() -> Lang {
    match CURRENT.load(Ordering::Relaxed) {
        1 => Lang::De,
        _ => Lang::En,
    }
}

type Catalog = HashMap<&'static str, &'static str>;

fn parse_catalog(src: &'static str, name: &str) -> Catalog {
    match toml::from_str::<HashMap<String, String>>(src) {
        Ok(map) => map
            .into_iter()
            .map(|(k, v)| (&*Box::leak(k.into_boxed_str()), &*Box::leak(v.into_boxed_str())))
            .collect(),
        Err(e) => {
            log::error!("Broken translation catalog {name}: {e}");
            HashMap::new()
        }
    }
}

fn catalog(lang: Lang) -> Option<&'static Catalog> {
    static DE: OnceLock<Catalog> = OnceLock::new();
    match lang {
        Lang::En => None,
        Lang::De => Some(DE.get_or_init(|| parse_catalog(include_str!("../i18n/de.toml"), "de.toml"))),
    }
}

/// Translate an English text into the current language.
pub fn tr(english: &'static str) -> &'static str {
    catalog(current()).and_then(|c| c.get(english).copied()).unwrap_or(english)
}

/// Replace `{name}` placeholders in a translated template.
pub fn fill(template: &str, args: &[(&str, String)]) -> String {
    let mut out = template.to_string();
    for (k, v) in args {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

/// `tr!("text")` → `&'static str`; `tr!("text {x}", x = value)` → `String`.
macro_rules! tr {
    ($s:literal) => {
        $crate::i18n::tr($s)
    };
    ($s:literal, $($k:ident = $v:expr),+ $(,)?) => {
        $crate::i18n::fill($crate::i18n::tr($s), &[$((stringify!($k), ($v).to_string())),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Decode the body of a Rust string literal (escapes and `\` line continuations).
    fn unescape(body: &str) -> String {
        let mut out = String::new();
        let mut it = body.chars().peekable();
        while let Some(c) = it.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('\n') => {
                    while it.peek().is_some_and(|c| c.is_whitespace()) {
                        it.next();
                    }
                }
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => {}
            }
        }
        out
    }

    /// All literals passed to `tr!` anywhere in `src/`.
    fn source_keys() -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "rs") || path.ends_with("i18n.rs") {
                continue;
            }
            // rustc reads CRLF as LF (sources are checked out with CRLF on
            // Windows); do the same so literals match.
            let src = std::fs::read_to_string(&path).unwrap().replace("\r\n", "\n");
            let mut rest = src.as_str();
            while let Some(i) = rest.find("tr!(") {
                // Skip identifiers that merely end in `tr` (e.g. `include_str!`).
                let prev = rest[..i].chars().next_back();
                rest = &rest[i + 4..];
                if prev.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                rest = rest.trim_start();
                let Some(body) = rest.strip_prefix('"') else { continue };
                rest = body;
                let mut end = 0;
                let b = rest.as_bytes();
                while b[end] != b'"' {
                    end += if b[end] == b'\\' { 2 } else { 1 };
                }
                keys.insert(unescape(&rest[..end]));
                rest = &rest[end..];
            }
        }
        keys
    }

    fn placeholders(s: &str) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        let mut rest = s;
        while let Some(i) = rest.find('{') {
            let Some(j) = rest[i..].find('}') else { break };
            set.insert(rest[i + 1..i + j].to_string());
            rest = &rest[i + j..];
        }
        set
    }

    #[test]
    fn german_catalog_is_complete_and_consistent() {
        let de = catalog(Lang::De).unwrap();
        let keys = source_keys();
        assert!(keys.len() > 100, "found only {} tr! strings", keys.len());
        let missing: Vec<_> = keys.iter().filter(|k| !de.contains_key(k.as_str())).collect();
        assert!(missing.is_empty(), "missing German translations: {missing:#?}");
        let unused: Vec<_> = de.keys().filter(|k| !keys.contains(**k)).collect();
        assert!(unused.is_empty(), "unused German entries: {unused:#?}");
        for (k, v) in de {
            assert_eq!(placeholders(k), placeholders(v), "placeholder mismatch in {k:?}");
        }
    }

    #[test]
    fn detects_tags() {
        assert_eq!(Lang::from_tag("de-DE"), Some(Lang::De));
        assert_eq!(Lang::from_tag("de_AT.UTF-8"), Some(Lang::De));
        assert_eq!(Lang::from_tag("en-US"), Some(Lang::En));
        assert_eq!(Lang::from_tag("fr-FR"), None);
    }

    #[test]
    fn fills_placeholders() {
        assert_eq!(fill("a {x} b {y}", &[("x", "1".into()), ("y", "2".into())]), "a 1 b 2");
    }
}
