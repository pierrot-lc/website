project('factory', 'c')

treesitterdep = dependency('tree-sitter', method : 'pkg-config')
grammarsdep = dependency('tree-sitter-grammars', method : 'pkg-config')

incdir = include_directories('include')
sources = files(
  'src/hash.c',
  'src/parsers/markdown.c',
  'src/parsers/markdown_inline.c',
  'src/parsers/yaml.c',
  'src/tree.c',
  'src/ts_utils.c',
  'src/writers/body.c',
  'src/writers/head.c',
)

executable('factory', 'main.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
executable('hash-codes', 'hash_codes.c', 'src/hash.c', include_directories : incdir)

test_body = executable('test-body', 'tests/test_body.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('blockquote', test_body, args: ['../tests/body/blockquote.md', '../tests/body/blockquote.html'])
test('code-block', test_body, args: ['../tests/body/code_block.md', '../tests/body/code_block.html'])
test('emph', test_body, args: ['../tests/body/emph.md', '../tests/body/emph.html'])
test('heading', test_body, args: ['../tests/body/heading.md', '../tests/body/heading.html'])
test('image', test_body, args: ['../tests/body/img.md', '../tests/body/img.html'])
test('latex', test_body, args: ['../tests/body/latex.md', '../tests/body/latex.html'])
test('link', test_body, args: ['../tests/body/link.md', '../tests/body/link.html'])
test('list', test_body, args: ['../tests/body/list.md', '../tests/body/list.html'])
test('page-info', test_body, args: ['../tests/body/page_info.md', '../tests/body/page_info.html'])
test('par', test_body, args: ['../tests/body/paragraph.md', '../tests/body/paragraph.html'])

test_head = executable('test-head', 'tests/test_head.c', sources, include_directories : incdir, dependencies : [treesitterdep, grammarsdep])
test('commons', test_head, args: ['../tests/head/commons.md', '../tests/head/commons.html'])
test('css', test_head, args: ['../tests/head/css.md', '../tests/head/css.html'])
test('script', test_head, args: ['../tests/head/script.md', '../tests/head/script.html'])
