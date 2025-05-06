#ifndef PARSERS_UTILS_H
#define PARSERS_UTILS_H

#include <tree_sitter/api.h>

TSTree *parse(const char *, const TSLanguage *);

#endif
