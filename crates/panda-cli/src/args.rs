//! A small, dependency-free command-line option parser shared by every subcommand.

use std::collections::HashSet;

use crate::CliError;

/// Parsed command-line arguments of one subcommand.
#[derive(Debug, Default)]
pub(crate) struct Args {
    pub(crate) positional: Vec<String>,
    /// `(option name without dashes, value)` in order.
    pub(crate) values: Vec<(String, String)>,
    pub(crate) flags: HashSet<String>,
}

impl Args {
    /// Parse `args`. `valued` options take a value (`--opt V` or `--opt=V`); `flags` do not.
    /// Short aliases are given as `("v", "verbose")` pairs.
    pub(crate) fn parse(args: &[String], valued: &[&str], flags: &[&str], short: &[(&str, &str)]) -> Result<Args, CliError> {
        let mut out = Args::default();
        let mut it = args.iter();
        let mut only_positional = false;
        while let Some(a) = it.next() {
            if only_positional || a == "-" || !a.starts_with('-') {
                out.positional.push(a.clone());
                continue;
            }
            if a == "--" {
                only_positional = true;
                continue;
            }
            let body = if let Some(long) = a.strip_prefix("--") {
                long.to_string()
            } else {
                let s = &a[1..];
                short
                    .iter()
                    .find(|(k, _)| *k == s)
                    .map(|(_, long)| long.to_string())
                    .ok_or_else(|| CliError::Usage(format!("unknown option '{a}'")))?
            };
            let (name, inline) = match body.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (body, None),
            };
            if valued.contains(&name.as_str()) {
                let v = match inline {
                    Some(v) => v,
                    None => it.next().cloned().ok_or_else(|| CliError::Usage(format!("option '--{name}' needs a value")))?,
                };
                out.values.push((name, v));
            } else if flags.contains(&name.as_str()) {
                if inline.is_some() {
                    return Err(CliError::Usage(format!("option '--{name}' takes no value")));
                }
                out.flags.insert(name);
            } else {
                return Err(CliError::Usage(format!("unknown option '{a}'")));
            }
        }
        Ok(out)
    }

    pub(crate) fn all(&self, name: &str) -> impl Iterator<Item = &str> {
        self.values.iter().filter(move |(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    pub(crate) fn last(&self, name: &str) -> Option<&str> {
        self.all(name).last()
    }

    pub(crate) fn u64(&self, name: &str) -> Result<Option<u64>, CliError> {
        self.last(name)
            .map(|v| {
                v.replace('_', "")
                    .parse::<u64>()
                    .map_err(|_| CliError::Usage(format!("--{name}: expected a non-negative integer, got '{v}'")))
            })
            .transpose()
    }

    pub(crate) fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// Exactly `names.len()` positionals.
    pub(crate) fn positionals(&self, cmd: &str, names: &[&str]) -> Result<&[String], CliError> {
        if self.positional.len() != names.len() {
            let want = names.iter().map(|n| format!("<{n}>")).collect::<Vec<_>>().join(" ");
            return Err(CliError::Usage(format!(
                "'panda {cmd}' expects {want}, got {} argument(s) (see 'panda help')",
                self.positional.len()
            )));
        }
        Ok(&self.positional)
    }
}

impl Args {
    /// `--name` as a `u32`.
    pub(crate) fn u32(&self, name: &str) -> Result<Option<u32>, CliError> {
        self.u64(name)?
            .map(|v| u32::try_from(v).map_err(|_| CliError::Usage(format!("--{name}: {v} is too large"))))
            .transpose()
    }

    /// `--name` as a positive floating-point number.
    pub(crate) fn positive_f64(&self, name: &str) -> Result<Option<f64>, CliError> {
        self.last(name)
            .map(|v| match v.parse::<f64>() {
                Ok(x) if x.is_finite() && x > 0.0 => Ok(x),
                _ => Err(CliError::Usage(format!("--{name}: expected a positive number, got '{v}'"))),
            })
            .transpose()
    }

    /// `--name` restricted to one of `choices` (case-insensitive; returns the canonical choice).
    pub(crate) fn choice(&self, name: &str, choices: &[&'static str]) -> Result<Option<&'static str>, CliError> {
        self.last(name)
            .map(|v| {
                choices.iter().copied().find(|c| c.eq_ignore_ascii_case(v)).ok_or_else(|| {
                    CliError::Usage(format!("--{name}: expected one of {}, got '{v}'", choices.join(", ")))
                })
            })
            .transpose()
    }

    /// Between `min` and `max` positionals.
    pub(crate) fn positionals_between(&self, cmd: &str, min: usize, max: usize, usage: &str) -> Result<&[String], CliError> {
        if self.positional.len() < min || self.positional.len() > max {
            return Err(CliError::Usage(format!(
                "'panda {cmd}' expects {usage}, got {} argument(s) (see 'panda help')",
                self.positional.len()
            )));
        }
        Ok(&self.positional)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn arg_parsing() {
        let a = Args::parse(
            &s(&["d.json", "--ticks", "5", "--set=A=1", "--set", "B=2", "-v", "--", "--x"]),
            &["ticks", "set"],
            &["verbose"],
            &[("v", "verbose")],
        )
        .unwrap();
        assert_eq!(a.positional, s(&["d.json", "--x"]));
        assert_eq!(a.u64("ticks").unwrap(), Some(5));
        assert_eq!(a.all("set").collect::<Vec<_>>(), ["A=1", "B=2"]);
        assert!(a.flag("verbose"));
        assert!(Args::parse(&s(&["--nope"]), &[], &[], &[]).is_err());
        assert!(Args::parse(&s(&["--ticks"]), &["ticks"], &[], &[]).is_err());
        assert!(Args::parse(&s(&["--json=1"]), &[], &["json"], &[]).is_err());
        let a = Args::parse(&s(&["--ticks", "abc"]), &["ticks"], &[], &[]).unwrap();
        assert!(matches!(a.u64("ticks"), Err(CliError::Usage(_))));
    }
}
