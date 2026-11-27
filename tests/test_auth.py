import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from core.auth import authenticate_user, register_user


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / 'users.db'
        self.env_patch = patch.dict(os.environ, {'POLICYAI_DB_PATH': str(self.db_path)}, clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.tmpdir.cleanup()

    def test_register_and_authenticate_user(self):
        created = register_user('Alex Morgan', 'Alex@Example.com', 'SecurePass123', 'SecurePass123')
        self.assertEqual(created['name'], 'Alex Morgan')
        self.assertEqual(created['email'], 'alex@example.com')
        self.assertEqual(created['initials'], 'AM')
        self.assertNotIn('password_hash', created)
        self.assertNotIn('salt', created)

        authed = authenticate_user('  alex@example.com ', 'SecurePass123')
        self.assertEqual(authed['id'], created['id'])

    def test_reject_duplicate_and_invalid_inputs(self):
        register_user('Sam Lee', 'sam@example.com', 'StrongPass1', 'StrongPass1')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            register_user('Sam Lee', 'SAM@example.com', 'StrongPass1', 'StrongPass1')
        with self.assertRaisesRegex(ValueError, 'name'):
            register_user('A', 'a@example.com', 'StrongPass1', 'StrongPass1')
        with self.assertRaisesRegex(ValueError, 'email'):
            register_user('Valid Name', 'not-an-email', 'StrongPass1', 'StrongPass1')
        with self.assertRaisesRegex(ValueError, '8 characters'):
            register_user('Valid Name', 'v@example.com', 'short', 'short')
        with self.assertRaisesRegex(ValueError, 'match'):
            register_user('Valid Name', 'v@example.com', 'StrongPass1', 'Different1')

    def test_reject_wrong_credentials(self):
        register_user('Jordan Patel', 'jordan@example.com', 'PolicyPass99', 'PolicyPass99')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            authenticate_user('jordan@example.com', 'WrongPass99')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            authenticate_user('missing@example.com', 'PolicyPass99')


if __name__ == '__main__':
    unittest.main()
