#include <tree_sitter/api.h>

#include "convert_tree_md_inline.h"
#include "hash.h"
#include "tree.h"
#include "utils.h"

/*
 * *Main*
 *
 * This is the main function of the file.
 */

Node *convert_tree_md_inline(const char *source, TSTree *tree) {
  TSNode root_node = ts_tree_root_node(tree);
  char *content = node_text(source, root_node);
  return create_node(HASH_INLINE, content);
}
