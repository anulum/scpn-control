# Git path exposure inspection

`tools/check_history_exposure.py` compares tracked path names with its blocked
expressions and explicit allowed exceptions. It reads Git metadata and names;
it does not read file contents, fetch refs, delete files or rewrite history.

```bash
python tools/check_history_exposure.py --repo . --current-only --json
python tools/check_history_exposure.py --repo . --json
```

`--repo` must select the worktree root. A subdirectory refuses because Git can
otherwise limit index selection to that prefix. The default root is anchored to
the script, while an explicit relative path resolves from the caller's working
directory. Bare repositories are outside this index-plus-history interface.

`--current-only` inspects the index, including staged additions and indexed
files missing from disk. It does not inspect untracked or ignored files. The
default inspection also adds names from history reachable through local refs,
including names deleted from the current tree. Reflogs, unreachable objects
and remote-only refs are not inspected. Calls are sequential observations,
without a coherent index/ref snapshot.

Names are NUL-delimited by Git and decoded with UTF-8/surrogateescape. Whitespace,
Unicode, quotes and embedded control characters are retained rather than
interpreted as Git's display quoting. Python sorts the resulting literal names.
`find_first_commit` uses literal pathspecs, so brackets and wildcards in a name
cannot select another file. The existing `first_commit` field is the last ID
from Git's default all-ref log order without rename following; it does not
promise the earliest timestamp across branches. An index-only name has an empty
commit ID.

The public APIs are `collect_current_paths`, `collect_history_paths`,
`find_first_commit`, `collect_exposures`, `Exposure` and `main`. Individual
collectors propagate Git/IO errors. Their `repo` parameter supplies Git's working
context; use the worktree root for complete inspection. `collect_exposures`
requires that root, combines the selected sets, and applies the unchanged
case-sensitive blocked/allowed expressions. A passing result means no matching
names were observed; it does not certify absence of secret contents or remote
publication.

The CLI returns 0 when no blocked names are observed, 1 for exposure findings
and 2 for an operational refusal. Operational errors use fixed stderr and emit
no success JSON. Argparse help and invalid options retain exits 0 and 2. JSON
escapes literal names as strings; text output escapes nonprintable names onto
one line. The existing path patterns and allow exceptions remain authoritative
in the owning Python module.

The owner tests use actual Git index entries, object trees, commits and local
refs. Git plumbing supplies historical control-character names without requiring
platform-restricted working filenames. They cover deleted names, staged Unicode,
whitespace preservation, literal path lookup, actual missing Git/worktrees,
root-selection refusals and cold command output. They do not mock Git output.
