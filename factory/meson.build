project('factory', 'c')

treesitterdep = dependency('tree-sitter', method : 'pkg-config')
grammarsdep = dependency('tree-sitter-grammars', method : 'pkg-config')

incdir = include_directories('include')
sources = files(
  'src/convert.c',
  'src/hash.c',
  'src/utils.c',
)

executable('factory', 'main.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
executable('hash-codes', 'hash_codes.c', 'src/hash.c', include_directories : incdir)

test_convert = executable('test-convert', 'tests/test_convert.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('header', test_convert, args: ['../tests/examples/header.md', '../tests/examples/header.html', '../tests/outs/header.html'])
test('par', test_convert, args: ['../tests/examples/par.md', '../tests/examples/par.html', '../tests/outs/par.html'])
test('emph', test_convert, args: ['../tests/examples/emph.md', '../tests/examples/emph.html', '../tests/outs/emph.html'])
