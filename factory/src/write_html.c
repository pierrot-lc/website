#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "hash.h"
#include "tree.h"
#include "write_html.h"

/* Place a pair of opening and closing balise and write the children in the
 * middle.
 */
static void _balise(FILE *, Node *, char *);

/* Write all children of the given node.
 */
static void _children(FILE *, Node *);

/*
 * *Writers*
 *
 * Each function here write some HTML code based on the specific Node type.
 */

/*
 * *Utils*
 */

static void _balise(FILE *file, Node *node, char *balise) {
  fprintf(file, "<%s>", balise);
  _children(file, node);
  fprintf(file, "</%s>", balise);
}

static void _children(FILE *file, Node *node) {
  for (int i = 0; i < node->child_count; i++)
    write_html(file, node->children[i]);
}

/*
 * *Main*
 */

void write_html(FILE *file, Node *root) {
  switch (root->code) {
  case HASH_ATX_HEADING:
    _balise(file, root, root->content);
    fprintf(file, "\n");
    break;

  case HASH_EMPH_EM:
    _balise(file, root, "em");
    break;

  case HASH_EMPH_STRONG:
    _balise(file, root, "strong");
    break;

  case HASH_PARAGRAPH:
    _balise(file, root, "p");
    fprintf(file, "\n");
    break;

  case HASH_TEXT:
    fprintf(file, "%s", root->content);
    break;

  case HASH_DOCUMENT:
  case HASH_INLINE:
  case HASH_SECTION:
    _children(file, root);
    break;

  default:
    fprintf(stderr, "[HTML] Unknown hash: %u\n", root->code);
    assert(false);
  }
}
