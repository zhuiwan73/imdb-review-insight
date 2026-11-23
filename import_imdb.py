"""Import original Stanford aclImdb reviews into the movie-scoped corpus.

Source: https://ai.stanford.edu/~amaas/data/sentiment/
The archive's URLs identify IMDb movie pages, and the JSONL supplies local
review IDs that can be used to show evidence from the imported text.
"""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import tarfile
import tempfile


SPLITS = {"train": ("train",), "test": ("test",), "all": ("train", "test")}
URL_MEMBER = re.compile(r"(?:aclImdb/)?(train|test)/urls_(pos|neg)\.txt")
REVIEW_MEMBER = re.compile(r"(?:aclImdb/)?(train|test)/(pos|neg)/([0-9]+)_[0-9]+\.txt")
MOVIE_PATH = re.compile(r"/title/(tt[0-9]{7,})(?=[/?#]|$)")
LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def read_utf8(archive, member):
    with archive.extractfile(member) as file:
        return file.read().decode("utf-8-sig")


def movie_urls(archive_path, splits):
    urls = {}
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            match = URL_MEMBER.fullmatch(member.name.removeprefix("./"))
            if not member.isfile() or not match or match[1] not in splits:
                continue
            split, polarity = match.groups()
            key = split, polarity
            if key in urls:
                raise ValueError(f"Duplicate URL list for {split}/{polarity}.")
            movie_ids = []
            for line_number, url in enumerate(read_utf8(archive, member).splitlines(), 1):
                movie = MOVIE_PATH.search(url)
                if not movie:
                    raise ValueError(f"Invalid movie URL in {member.name}, line {line_number}.")
                movie_ids.append(movie[1])
            urls[key] = movie_ids

    missing = {f"{split}/urls_{polarity}.txt" for split in splits for polarity in ("pos", "neg")
               if (split, polarity) not in urls}
    if missing:
        raise ValueError(f"Archive is missing {', '.join(sorted(missing))}.")
    return urls


def ranked_movies(urls):
    positive = Counter(movie for (_, label), movies in urls.items() if label == "pos" for movie in movies)
    negative = Counter(movie for (_, label), movies in urls.items() if label == "neg" for movie in movies)
    ranked = sorted(positive.keys() | negative.keys(), key=lambda movie: (
        -min(positive[movie], negative[movie]),
        -(positive[movie] + negative[movie]),
        movie,
    ))
    return [(movie, positive[movie], negative[movie]) for movie in ranked]


def read_titles(path):
    if path is None:
        return {}
    titles = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(titles, dict) or any(not isinstance(value, str) for value in titles.values()):
        raise ValueError("Title lookup must be a JSON object mapping movie IDs to title strings.")
    return titles


def import_reviews(archive_path, urls, selected_movies, titles):
    expected = {
        (split, polarity, review_index): movie
        for (split, polarity), movies in urls.items()
        for review_index, movie in enumerate(movies)
        if movie in selected_movies
    }
    records = []
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            match = REVIEW_MEMBER.fullmatch(member.name.removeprefix("./"))
            if not member.isfile() or not match:
                continue
            split, polarity, review_index = match.groups()
            movie = expected.pop((split, polarity, int(review_index)), None)
            if movie is None:
                continue
            text = html.unescape(LINE_BREAK.sub("\n", read_utf8(archive, member)))
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if not text:
                raise ValueError(f"Empty review at {member.name}.")
            records.append({
                "movie_id": movie,
                "title": titles.get(movie, "").strip() or f"IMDb {movie}",
                "review_id": f"{split}-{polarity}-{int(review_index)}",
                "text": text,
            })
    if expected:
        split, polarity, review_index = next(iter(expected))
        raise ValueError(f"Archive is missing {len(expected)} selected reviews, including {split}-{polarity}-{review_index}.")
    return sorted(records, key=lambda row: (row["movie_id"], row["review_id"]))


def write_jsonl(output, records):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the original aclImdb_v1.tar.gz archive.")
    parser.add_argument("output", type=Path, nargs="?", help="Destination corpus JSONL.")
    parser.add_argument("--split", choices=SPLITS, default="test", help="Archive split to import (default: test).")
    parser.add_argument("--movie-id", action="append", default=[], help="Select a movie ID (repeatable).")
    parser.add_argument("--max-movies", type=int, default=20, help="How many movies to choose by default.")
    parser.add_argument("--titles", type=Path, help="Optional JSON movie ID to display title lookup.")
    parser.add_argument("--list-movies", action="store_true", help="List selected movie IDs and review counts as TSV.")
    args = parser.parse_args()
    if args.max_movies < 1:
        parser.error("--max-movies must be positive.")
    if not args.list_movies and args.output is None:
        parser.error("Provide an output path or use --list-movies.")
    if args.output is not None and args.archive.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the input archive.")

    try:
        urls = movie_urls(args.archive, SPLITS[args.split])
        ranked = ranked_movies(urls)
        if args.movie_id:
            missing = set(args.movie_id) - {movie for movie, _, _ in ranked}
            if missing:
                raise ValueError(f"Movie IDs not found in {args.split}: {', '.join(sorted(missing))}.")
            selected = [row for row in ranked if row[0] in args.movie_id]
        else:
            selected = ranked[:args.max_movies]
        if not selected:
            raise ValueError("The selected split is empty.")

        if args.list_movies:
            print("movie_id\tpositive_reviews\tnegative_reviews")
            for row in selected:
                print(*row, sep="\t")
            return

        reviews = import_reviews(args.archive, urls, {row[0] for row in selected}, read_titles(args.titles))
        write_jsonl(args.output, reviews)
        print(f"Imported {len(reviews)} reviews across {len(selected)} movies into {args.output}.")
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Import failed: {error}\n")


if __name__ == "__main__":
    main()
