#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "hash.h"
#include "parsers/markdown.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "writers/body.h"

/**
 * Write all children of the given node.
 */
static void write_children(FILE *file, Node *node) {
  for (int i = 0; i < node->child_count; i++)
    write_tree(file, node->children[i]);
}

/**
 * Place a pair of opening and closing balise and write the children in the
 * middle.
 */
static void balise(FILE *file, Node *node, char *type) {
  fprintf(file, "<%s>", type);
  write_children(file, node);
  fprintf(file, "</%s>", type);
}

static void alink(FILE *file, Node *node) {
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
    write_children(file, text);

  fprintf(file, "</a>");
}

static void code_block(FILE *file, Node *node) {
  if (node->content != NULL)
    fprintf(file, "<pre><code class=\"language-%s\">", node->content);
  else
    fprintf(file, "<pre><code>");
  fprintf(file, "%s", node->children[0]->content);
  fprintf(file, "</code></pre>\n");
}

static void image(FILE *file, Node *node) {
  Node *child;
  Node *link = NULL, *desc = NULL;

  for (int i = 0; i < node->child_count; i++) {
    child = node->children[i];

    switch (child->code) {
    case HASH_IMAGE_DESCRIPTION:
      desc = child;
      break;

    case HASH_LINK_DESTINATION:
      link = child;
      break;

    case HASH_LINK_LABEL:
      link = search_label_destination(tree_root(node), child->content);
      break;
    }
  }

  assert(link != NULL);

  if (desc != NULL)
    fprintf(file, "<img src=\"%s\" alt=\"%s\">", link->content, desc->content);
  else
    fprintf(file, "<img src=\"%s\">", link->content);
}

static void latex(FILE *file, Node *node) {
  if (node->code == HASH_LATEX_DISPLAY)
    fprintf(file, "<span class=\"latex-display\">%s</span>", node->content);
  else if (node->code == HASH_LATEX_INLINE)
    fprintf(file, "<span class=\"latex-inline\">%s</span>", node->content);
  else {
    assert(false);
  }
}

static void list(FILE *file, Node *node) {
  fprintf(file, "<ul>\n");
  for (int i = 0; i < node->child_count; i++) {
    balise(file, node->children[i], "li");
    fprintf(file, "\n");
  }
  fprintf(file, "</ul>\n");
}

void write_tree(FILE *file, Node *node) {
  switch (node->code) {
  case HASH_ATX_HEADING:
    balise(file, node, node->content);
    fprintf(file, "\n");
    break;

  case HASH_BLOCK_QUOTE:
    fprintf(file, "<blockquote>\n");
    write_children(file, node);
    fprintf(file, "</blockquote>\n");
    break;

  case HASH_CODE_SPAN:
    balise(file, node, "code");
    break;

  case HASH_EMPHASIS:
    balise(file, node, "em");
    break;

  case HASH_FENCED_CODE_BLOCK:
    code_block(file, node);
    break;

  case HASH_IMAGE:
    image(file, node);
    break;

  case HASH_LATEX_DISPLAY:
  case HASH_LATEX_INLINE:
    latex(file, node);
    break;

  case HASH_LINK:
    alink(file, node);
    break;

  case HASH_LIST:
    list(file, node);
    break;

  case HASH_PARAGRAPH:
    balise(file, node, "p");
    fprintf(file, "\n");
    break;

  case HASH_STRONG_EMPHASIS:
    balise(file, node, "strong");
    break;

  case HASH_HTML_BLOCK:
  case HASH_TEXT:
    fprintf(file, "%s", node->content);
    break;

  case HASH_THEMATIC_BREAK:
    fprintf(file, "<hr />\n");
    break;

  case HASH_DOCUMENT:
  case HASH_INLINE:
  case HASH_SECTION:
    write_children(file, node);
    break;

  case HASH_BLOCK_CONTINUATION:
  case HASH_BLOCK_QUOTE_MARKER:
  case HASH_LINK_REFERENCE_DEFINITION:
  case HASH_MINUS_METADATA:
  case HASH_STREAM:
    break;

  default:
    fprintf(stderr, "[BODY-MAIN] Unknown hash: %u\n", node->code);
    assert(false);
  }
}

void write_page_info(FILE *file, Node *tree) {
  Node *date, *tags, *title, *illustration;

  if ((illustration = get_key(tree, "illustration")) != NULL)
    fprintf(file, "<img src=\"%s\" class=\"article-illustration\">\n",
            get_value_scalar(illustration));

  if ((title = get_key(tree, "title")) != NULL)
    fprintf(file, "<h1 class=\"article-title\">%s</h1>\n",
            get_value_scalar(title));

  tags = get_key(tree, "tags");
  date = get_key(tree, "date");
  if (tags != NULL || date != NULL) {
    fprintf(file, "<ul class=\"article-tags\">\n");

    if (date != NULL)
      fprintf(file, "<li>%s</li>\n", get_value_scalar(date));

    if (tags != NULL)
      for (int i = 0; i < tags->child_count; i++) {
        assert(tags->children[i]->code == HASH_BLOCK_SEQUENCE_ITEM);
        assert(tags->children[i]->children[0] != NULL);
        fprintf(file, "<li>%s</li>\n", tags->children[i]->children[0]->content);
      }

    fprintf(file, "</ul>\n");
  }
}
