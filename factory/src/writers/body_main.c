#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "hash.h"
#include "parsers/markdown.h"
#include "tree.h"
#include "writers/body_main.h"

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

static void _link(FILE *file, Node *node) {
  Node *child;
  Node *text = NULL, *destination = NULL;

  for (int i = 0; i < node->child_count; i++) {
    child = node->children[i];
    switch (child->code) {
    case HASH_LINK_TEXT:
      text = child;
      break;

    case HASH_LINK_DESTINATION:
      destination = child;
      break;

    case HASH_LINK_LABEL:
      destination = search_label_destination(tree_root(node), child->content);
      break;

    default:
      fprintf(stderr, "[WRITER LINK] Unexpected hash type: %u", child->code);
      assert(false);
    }
  }

  if (destination != NULL)
    fprintf(file, "<a href=\"%s\">", destination->content);
  else
    fprintf(file, "<a>");

  if (text != NULL)
    _children(file, text);

  fprintf(file, "</a>");
}

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
    write_body_main(file, node->children[i]);
}

/*
 * *Main*
 */

void write_body_main(FILE *file, Node *node) {
  switch (node->code) {
  case HASH_ATX_HEADING:
    _balise(file, node, node->content);
    fprintf(file, "\n");
    break;

  case HASH_EMPH_EM:
    _balise(file, node, "em");
    break;

  case HASH_EMPH_STRONG:
    _balise(file, node, "strong");
    break;

  case HASH_DOCUMENT:
    fprintf(file, "<main>\n");
    _children(file, node);
    fprintf(file, "</main>\n");
    break;

  case HASH_LINK:
    _link(file, node);
    break;

  case HASH_PARAGRAPH:
    _balise(file, node, "p");
    fprintf(file, "\n");
    break;

  case HASH_TEXT:
    fprintf(file, "%s", node->content);
    break;

  case HASH_INLINE:
  case HASH_SECTION:
    _children(file, node);
    break;

  case HASH_LINK_REFERENCE_DEFINITION:
  case HASH_MINUS_METADATA:
  case HASH_STREAM:
    break;

  default:
    fprintf(stderr, "[BODY-MAIN] Unknown hash: %u\n", node->code);
    assert(false);
  }
}
