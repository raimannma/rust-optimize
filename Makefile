.PHONY: changelog

# Regenerate CHANGELOG.md from git tags using git-cliff.
# Install: cargo install git-cliff
changelog:
	git-cliff --output CHANGELOG.md
