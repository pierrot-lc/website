#ifndef TS_UTILS_H
#define TS_UTILS_H

#include <tree_sitter/api.h>

/**
 * Read the whole file provided its path. You are responsible to `free` the
 * returned string.
 */
char *read_file(const char *);

/**
 * Extract the part of text delimited by `start` and `end`. You are responsible
 * to `free` the returned string.
 */
char *extract_text(const char *, unsigned int, unsigned int);

/**
 * Read the node's text in the source code based on the starting and ending
 * bytes. You are responsible to `free` the returned string.
 */
char *ts_node_text(const char *, TSNode);

/**
 * Return the root node by iteratively going up the tree.
 */
TSNode ts_tree_root(TSNode);

/**
 * Return the first child with the given hash code. Return a null node if
 * nothing has been found.
 */
TSNode ts_search(const TSNode ts_node, unsigned int hash_code);

/**
 * Parse the source code using treesitter.
 */
TSTree *ts_parse(const char *source, const TSLanguage *language);

#endif
