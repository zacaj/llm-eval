# Hidden-helper-file experiment

Stock aider only adds `files.solution` from each exercise's `.meta/config.json`
to the chat. These two exercises keep source the solution depends on under
`files.editor` (java word-search: Pair/WordLocation; go paasio: interface.go),
so the model never sees them. The `full-*` copies are identical except those
files are moved into `solution`.

Named `full-*` (not `*-complete`) so `--keywords java/exercises/practice/word-search`
does not substring-match the variant. Mounted read-only via docker-compose.yml.
