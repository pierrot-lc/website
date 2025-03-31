#include <assert.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "parse.h"

TSTree *parse(const char *source, const TSLanguage *language) {
  TSParser *parser;
  TSTree *tree;

  parser = ts_parser_new();
  assert(parser != NULL);
  ts_parser_set_language(parser, language);
  tree = ts_parser_parse_string(parser, NULL, source, strlen(source));
  assert(tree != NULL);
  ts_parser_delete(parser);
  return tree;
}
