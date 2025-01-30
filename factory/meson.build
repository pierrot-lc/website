project('factory', 'c')

treesitterdep = dependency('tree-sitter', method : 'pkg-config')
grammarsdep = dependency('tree-sitter-grammars', method : 'pkg-config')
executable('test-json-parser', 'test-json-parser.c', dependencies : [treesitterdep, grammarsdep])
