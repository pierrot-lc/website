project('factory', 'c')

treesitterdep = dependency('tree-sitter', method : 'pkg-config')
grammarsdep = dependency('tree-sitter-grammars', method : 'pkg-config')

incdir = include_directories('include')
sources = files(
  'src/hash.c',
  'src/parsers/markdown.c',
  'src/parsers/markdown_inline.c',
  'src/parsers/utils.c',
  'src/parsers/yaml.c',
  'src/tree.c',
  'src/ts_utils.c',
  'src/writers/body_main.c',
  'src/writers/head.c',
  'src/writers/header.c',
  'src/writers/html.c',
  'src/writers/utils.c',
)

executable('factory', 'main.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
executable('hash-codes', 'hash_codes.c', 'src/hash.c', include_directories : incdir)

test_body_main = executable('test-body-main', 'tests/test_body_main.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('header', test_body_main, args: ['../tests/body_main/header.md', '../tests/body_main/header.html'])
test('par', test_body_main, args: ['../tests/body_main/paragraph.md', '../tests/body_main/paragraph.html'])
test('emph', test_body_main, args: ['../tests/body_main/emph.md', '../tests/body_main/emph.html'])
test('latex', test_body_main, args: ['../tests/body_main/latex.md', '../tests/body_main/latex.html'])
test('link', test_body_main, args: ['../tests/body_main/link.md', '../tests/body_main/link.html'])
test('list', test_body_main, args: ['../tests/body_main/list.md', '../tests/body_main/list.html'])

test_head = executable('test-head', 'tests/test_head.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('css', test_head, args: ['../tests/head/css.md', '../tests/head/css.html'])
test('script', test_head, args: ['../tests/head/script.md', '../tests/head/script.html'])
test('title', test_head, args: ['../tests/head/title.md', '../tests/head/title.html'])

test_header = executable('test-header', 'tests/test_header.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('nav', test_header, args: ['../tests/header/nav.yaml', '../tests/header/nav.html'])
