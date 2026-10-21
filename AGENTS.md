# Working in this repository

## Tests

Do not write tests unless they were explicitly asked for. No new test files, and no new test
functions added to existing files.

The exception is a test that the change *breaks*: when behaviour changes and an existing assertion
no longer describes it, update that assertion so the suite keeps passing, and say so.

## The work log

Always record completed work in `local/DONE.md`, as a new numbered section following the format
already there: what the problem was, `### Changes Made:`, `### Design decisions:` and
`### Known limitations:`. State plainly what was and was not verified.

`local/` is untracked (see `.git/info/exclude`), so the log never enters a commit. Add to it anyway,
as part of finishing the work rather than as a separate request.
