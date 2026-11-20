import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.convert_data_splits import convert, verify_output


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'data_splits.zip'
        self.files = {
            'data_splits/dlbcl_train_fold_0.csv': 'slide,label\na,0\nb,1\n',
            'data_splits/dlbcl_val_fold_0.csv': 'slide,label\nc,1\nd,0\n',
            'data_splits/dlbcl_train_fold_1.csv': 'slide,label\nc,1\nd,0\n',
            'data_splits/dlbcl_val_fold_1.csv': 'slide,label\na,0\nb,1\n',
            'data_splits/ood_test.csv': 'slide,label,site\nx,1,elsewhere\n',
        }

    def write_source(self):
        with zipfile.ZipFile(self.source, 'w') as archive:
            for name, content in self.files.items():
                archive.writestr(name, content)

    def rows(self, archive, name):
        return list(csv.reader(io.StringIO(archive.read(name).decode('utf-8'))))

    def test_output_and_original_preserved(self):
        self.write_source()
        original = self.source.read_bytes()
        result = convert(self.source)
        self.assertEqual(self.source.read_bytes(), original)
        verify_output(self.source, result)
        with zipfile.ZipFile(result) as archive:
            self.assertEqual(self.rows(archive, 'cv/dataset.csv'),
                             [['slide_id', 'label'], ['a', '0'], ['b', '1'], ['c', '1'], ['d', '0']])
            self.assertEqual(self.rows(archive, 'cv/splits_0_bool.csv'),
                             [['slide_id', 'train', 'val', 'test'], ['a', 'True', 'False', 'False'],
                              ['b', 'True', 'False', 'False'], ['c', 'False', 'True', 'True'],
                              ['d', 'False', 'True', 'True']])
            self.assertEqual(self.rows(archive, 'cv/splits_0_descriptor.csv'),
                             [['label', 'train', 'val', 'test'], ['0', '1', '1', '1'], ['1', '1', '1', '1']])
            for fold in range(2):
                with self.subTest(fold=fold):
                    membership = self.rows(archive, f'cv/splits_{fold}_bool.csv')
                    for slide, train, val, test in membership[1:]:
                        self.assertEqual(val, test, slide)
                        self.assertEqual(train == 'True', val == 'False', slide)
                    descriptor = self.rows(archive, f'cv/splits_{fold}_descriptor.csv')
                    for label, train, val, test in descriptor[1:]:
                        self.assertEqual(val, test, label)
            self.assertEqual(self.rows(archive, 'test_sets/ood_test.csv'),
                             [['slide_id', 'label', 'site'], ['x', '1', 'elsewhere']])
        with self.assertRaises(ValueError):
            convert(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)

    def test_invalid_sources(self):
        cases = [
            ('dlbcl_train_fold_0.csv', 'slide,label\na,0\na,0\n'),
            ('dlbcl_train_fold_0.csv', 'slide,label\na,2\nb,1\n'),
            ('ood_test.csv', 'slide,label,site\na,1,elsewhere\n'),
            ('ood_test.csv', 'slide,label,site\na,0,elsewhere\n'),
            ('dlbcl_val_fold_0.csv', 'slide,label\na,0\nc,1\n'),
            ('dlbcl_train_fold_1.csv', 'slide,label\nc,1\n'),
            ('dlbcl_val_fold_1.csv', 'slide,label\nc,1\nd,0\n'),
        ]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content):
                key = 'data_splits/' + filename
                previous = self.files[key]
                self.files[key] = content
                self.write_source()
                with self.assertRaises(ValueError):
                    convert(self.source)
                self.files[key] = previous

    def test_verifier_rejects_tampering(self):
        self.write_source()
        result = convert(self.source)
        with zipfile.ZipFile(result, 'a') as archive:
            archive.writestr('unexpected', 'bad')
        with self.assertRaises(ValueError):
            verify_output(self.source, result)


if __name__ == '__main__':
    unittest.main()
