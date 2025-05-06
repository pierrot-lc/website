#ifndef CONVERT_TREE_MD_H
#define CONVERT_TREE_MD_H

#include <tree_sitter/api.h>

#include "tree.h"

/* Main function, used to convert the content of the markdown tree into the
 * given file.
 */
Node *convert_tree_md(const char *);

#endif
