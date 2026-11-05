"""Labelled matches from other games (public datasets), read into the same shape as our match.

    from mtah.external import dfl
    m = dfl.load("DFL-MAT-J03WMX")        # one Bundesliga match, parsed once into a cache

These are other matches, not BAR-ATH, so their labels may be used freely (RULES.md / FAQ.md:
any data about other games is allowed). They are downloaded by scripts/data/download_external.sh
into external/ (gitignored). See claude_notes/external_data.md for what each dataset holds.
"""
