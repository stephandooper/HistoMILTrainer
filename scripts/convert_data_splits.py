"""Convert the supplied split archive to the trainer's split-file layout."""
import argparse
import csv
import io
import os
from pathlib import Path
import re
import tempfile
import zipfile
from collections import Counter

FOLD = re.compile(r'^dlbcl_(train|val)_fold_(\d+)\.csv$')
README = """Converted data splits

Created with: python scripts/convert_data_splits.py INPUT [--output OUTPUT]
The CV dataset and boolean/descriptor split files are under cv/. No case_id is needed.
CV test membership intentionally matches validation membership for each fold, allowing
the trainer's final extra pass. This is not an independent evaluation. Do not run
histomil-splits: it would regenerate folds. All eight external test cohorts are retained
separately under test_sets/, including OOD metadata. For independent evaluation, run
prediction separately for each saved fold checkpoint and test CSV, then join predictions
to labels by slide_id to calculate metrics; the current predictor does not calculate them.
Feature files must be named
<slide_id without extension>.h5 and supplied through the appropriate feature directory.
In the supplied DLBCL archive, joint_2_test combines internal_test, hovon_test and
s05_test, while joint_1_test combines joint_2_test and stanford_test. These cohorts
overlap and are retained separately. Stanford contains only one positive slide.
"""


def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def read_source(path):
    folds, tests, labels, dataset = {}, {}, {}, {}
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        for name in names:
            base = Path(name).name
            match = FOLD.fullmatch(base)
            if not match and not base.endswith('_test.csv'):
                continue
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
                fields = reader.fieldnames
                if not fields or len(fields) != len(set(fields)) or 'slide' not in fields or 'label' not in fields or 'slide_id' in fields:
                    raise ValueError(f'{name}: expected distinct slide and label columns')
                rows, seen = [], set()
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f'{name}: malformed CSV row')
                    slide, label = row['slide'], row['label']
                    if not slide or slide in seen:
                        raise ValueError(f'{name}: empty or duplicate slide ID {slide!r}')
                    if label not in ('0', '1'):
                        raise ValueError(f'{name}: non-binary label {label!r}')
                    if slide in labels and labels[slide] != label:
                        raise ValueError(f'{name}: conflicting label for {slide!r}')
                    seen.add(slide)
                    labels[slide] = label
                    rows.append(row)
            if match:
                key = (int(match.group(2)), match.group(1))
                if key in folds:
                    raise ValueError(f'duplicate fold file: {key}')
                folds[key] = rows
                for row in rows:
                    dataset.setdefault(row['slide'], row['label'])
            else:
                if base in tests:
                    raise ValueError(f'duplicate test file: {base}')
                tests[base] = (fields, rows)
    indices = sorted({i for i, _ in folds})
    if not indices or indices != list(range(len(indices))):
        raise ValueError('fold indices must be consecutive from zero')
    pool = None
    validation_counts = Counter()
    for i in indices:
        if (i, 'train') not in folds or (i, 'val') not in folds:
            raise ValueError(f'fold {i}: missing train or val file')
        train, val = folds[i, 'train'], folds[i, 'val']
        train_ids = {r['slide'] for r in train}
        val_ids = {r['slide'] for r in val}
        if train_ids & val_ids:
            raise ValueError(f'fold {i}: train/val overlap')
        fold_pool = train_ids | val_ids
        if pool is None:
            pool = fold_pool
        elif pool != fold_pool:
            raise ValueError(f'fold {i}: inconsistent pool')
        validation_counts.update(val_ids)
    if not pool or any(validation_counts[slide] != 1 for slide in pool):
        raise ValueError('each CV slide must appear in validation exactly once')
    for name, (_, rows) in tests.items():
        if pool & {r['slide'] for r in rows}:
            raise ValueError(f'{name}: test/CV overlap')
    return indices, folds, tests, dataset


def expected_files(path):
    indices, folds, tests, dataset = read_source(path)
    files = {'README.txt': README.encode('utf-8'),
             'cv/dataset.csv': csv_bytes(['slide_id', 'label'], dataset.items())}
    for i in indices:
        train, val = folds[i, 'train'], folds[i, 'val']
        rows = [(r['slide'], 'True', 'False', 'False') for r in train]
        rows += [(r['slide'], 'False', 'True', 'True') for r in val]
        files[f'cv/splits_{i}_bool.csv'] = csv_bytes(['slide_id', 'train', 'val', 'test'], rows)
        counts = [Counter(r['label'] for r in group) for group in (train, val)]
        files[f'cv/splits_{i}_descriptor.csv'] = csv_bytes(
            ['label', 'train', 'val', 'test'], [(label, counts[0][label], counts[1][label], counts[1][label]) for label in ('0', '1')])
    for name, (fields, rows) in tests.items():
        files['test_sets/' + name] = csv_bytes(
            ['slide_id' if field == 'slide' else field for field in fields],
            [[row[field] for field in fields] for row in rows])
    return files


def verify_output(source, output):
    expected = expected_files(source)
    with zipfile.ZipFile(output) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('output archive has unexpected or duplicate members')
        for name, content in expected.items():
            if archive.read(name) != content:
                raise ValueError(f'output does not match source: {name}')


def convert(source, output=None):
    source = Path(source)
    output = Path(output) if output is not None else source.with_name('data_splits_converted.zip')
    if source.resolve() == output.resolve():
        raise ValueError('input and output must be different files')
    files = expected_files(source)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.splits-', suffix='.zip', delete=False) as tmp:
            temporary = Path(tmp.name)
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        verify_output(source, temporary)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        output = convert(args.input, args.output)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(f'Created {output}')


if __name__ == '__main__':
    main()
