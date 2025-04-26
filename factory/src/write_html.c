#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "hash.h"
#include "tree.h"
#include "write_html.h"

static void _children(FILE *, Node *);

/*
 * *Writers*
 *
 * Each function here write some HTML code based on the specific Node type.
 */

static void _heading(FILE *file, Node *node) {
  Node *title = node->children[0];

  fprintf(file, "<%s>", node->content);
  write_html(file, title);
  fprintf(file, "</%s>\n", node->content);
}

static void _inline(FILE *file, Node *node) {
  fprintf(file, "%s", node->content);
}

static void _paragraph(FILE *file, Node *node) {
  fprintf(file, "<p>");
  _children(file, node);
  fprintf(file, "</p>\n");
}

/*
 * *Utils*
 */

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
    _heading(file, root);
    break;

  case HASH_INLINE:
    _inline(file, root);
    break;

  case HASH_PARAGRAPH:
    _paragraph(file, root);
    break;

  case HASH_DOCUMENT:
  case HASH_SECTION:
    _children(file, root);
    break;

  default:
    fprintf(stderr, "Unknown hash: %u\n", root->code);
    assert(false);
  }
}
