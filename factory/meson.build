project('factory', 'c')

treesitterdep = dependency('tree-sitter', method : 'pkg-config')
grammarsdep = dependency('tree-sitter-grammars', method : 'pkg-config')

incdir = include_directories('include')
sources = files(
  'src/hash.c',
  'src/writer.c',
)

executable('parser', 'src/main.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
executable('hash-codes', 'src/hash_codes.c', sources, include_directories : incdir)

test_writer = executable('test-writer', 'tests/test_writer.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('test-header', test_writer, args: ['../tests/examples/header.typ', '../tests/examples/header.html', '../tests/outs/header.html'])
