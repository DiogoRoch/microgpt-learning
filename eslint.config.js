import eslint from '@eslint/js'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['tools/**/*.mjs'],
    languageOptions: { globals: { process: 'readonly', console: 'readonly' } },
  },
  {
    rules: {
      // Engine code passes Float32Arrays and readonly tuples around; these two
      // defaults produce more noise than signal there.
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },
)
