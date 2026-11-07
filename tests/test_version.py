import re
import unittest
from pathlib import Path

from tools.version import __version__

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.changelog = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")

    def test_version_is_semver(self):
        self.assertRegex(__version__, SEMVER)

    def test_changelog_keeps_unreleased_section_on_top(self):
        headings = re.findall(r"^## (.+)$", self.changelog, re.MULTILINE)
        self.assertEqual(headings[0], "[未发布]")

    def test_latest_release_matches_version(self):
        releases = re.findall(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$", self.changelog, re.MULTILINE)
        self.assertTrue(releases, "CHANGELOG has no released version heading")
        self.assertEqual(releases[0][0], __version__)

    def test_releases_are_in_descending_order(self):
        versions = [tuple(map(int, v.split("."))) for v in
                    re.findall(r"^## \[(\d+\.\d+\.\d+)\]", self.changelog, re.MULTILINE)]
        self.assertEqual(versions, sorted(versions, reverse=True))
        self.assertEqual(len(versions), len(set(versions)))


if __name__ == "__main__":
    unittest.main()
