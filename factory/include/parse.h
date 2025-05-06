#ifndef PARSE_H
#define PARSE_H

#include <tree_sitter/api.h>

TSTree *parse(const char *, const TSLanguage *);

#endif
