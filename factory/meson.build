project('factory', 'c')

treesitterdep = dependency('tree-sitter', method : 'pkg-config')
grammarsdep = dependency('tree-sitter-grammars', method : 'pkg-config')

incdir = include_directories('include')
sources = files(
  'src/convert_tree_md.c',
  'src/convert_tree_md_inline.c',
  'src/hash.c',
  'src/parse.c',
  'src/tree.c',
  'src/utils.c',
  'src/write_html.c',
)

executable('factory', 'main.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
executable('hash-codes', 'hash_codes.c', 'src/hash.c', include_directories : incdir)

test_convert = executable('test-factory', 'tests/test.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('header', test_convert, args: ['../tests/examples/header.md', '../tests/examples/header.html'])
test('par', test_convert, args: ['../tests/examples/paragraph.md', '../tests/examples/paragraph.html'])
test('emph', test_convert, args: ['../tests/examples/emph.md', '../tests/examples/emph.html'])
