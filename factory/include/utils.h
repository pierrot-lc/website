#ifndef UTILS_H
#define UTILS_H

#include <tree_sitter/api.h>

/* Read the whole file provided its path. You are responsible to `free` the
 * returned string.
 */
char *read_file(const char *);

/* Extract the part of text delimited by `start` and `end`. You are responsible
 * to `free` the returned string.
 */
char *extract_text(const char *, unsigned int, unsigned int);

/* Read the node's text in the source code based on the starting and ending
 * bytes. You are responsible to `free` the returned string.
 */
char *node_text(const char *, TSNode);

/* Return the root node by iteratively going up the tree.
 */
TSNode search_root(TSNode);

/* Search for the given hash value and node text within the node and its
 * children. Return the first node that match both criteria. In case of
 * failure, the returned node is the null node (`ts_node_is_null`).
 */
TSNode search_node(const char *, TSNode, unsigned int, char *);

#endif
