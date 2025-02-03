#ifndef UTILS_H
#define UTILS_H

#include <tree_sitter/api.h>

/* Read the whole file provided its path. You are responsible to `free` the
 * returned string.
 */
char *read_file(const char *);

/* Read the node's text in the source code based on the starting and ending
 * bytes. You are responsible to `free` the returned string.
 */
char *node_text(const char *, TSNode);

#endif
