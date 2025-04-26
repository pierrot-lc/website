#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "convert_tree_md.h"
#include "convert_tree_md_inline.h"
#include "hash.h"
#include "parse.h"
#include "tree.h"
#include "utils.h"

const TSLanguage *tree_sitter_markdown_inline(void);

static Node *_node(const char *, TSNode);
static void _children(Node *, const char *, TSNode);

/*
 * *Converters*
 *
 * Convert TSNode into Node. Each function is specifically made for a certain
 * type of TSNode.
 */

static Node *_heading(const char *source, TSNode ts_node) {
  assert(ts_node_named_child_count(ts_node) == 2);

  Node *node, *child;
  TSNode ts_marker, ts_content;
  char *content;

  const char *h1 = "h1";
  const char *h2 = "h2";
  const char *h;

  ts_marker = ts_node_child(ts_node, 0);
  switch (hash(ts_node_type(ts_marker))) {
  case HASH_ATX_H1_MARKER:
    h = h1;
    break;
  case HASH_ATX_H2_MARKER:
    h = h2;
    break;
  default:
    assert(false);
  }

  content = (char *)malloc(strlen(h) * sizeof(char));
  strcpy(content, h);
  node = create_node(HASH_ATX_HEADING, content);

  ts_content = ts_node_child(ts_node, 1);
  child = _node(source, ts_content);
  add_child(node, child);
  return node;
}

static Node *_inline(const char *source, TSNode ts_node) {
  Node *node;
  TSTree *tree;
  char *source_inline;

  source_inline = node_text(source, ts_node);
  tree = parse(source_inline, tree_sitter_markdown_inline());
  node = convert_tree_md_inline(source_inline, tree);
  free(source_inline);
  ts_tree_delete(tree);
  return node;
}

/*
 * *Utils*
 *
 * The following functions do not directly convert a TSNode into a Node.
 */

/* Choose the right convert function to use. */
static Node *_node(const char *source, TSNode ts_node) {
  Node *node;
  switch (hash(ts_node_type(ts_node))) {
  case HASH_ATX_HEADING:
    node = _heading(source, ts_node);
    break;

  case HASH_INLINE:
    node = _inline(source, ts_node);
    break;

  case HASH_PARAGRAPH:
    node = create_node(HASH_PARAGRAPH, NULL);
    _children(node, source, ts_node);
    break;

  case HASH_SECTION:
    node = create_node(HASH_SECTION, NULL);
    _children(node, source, ts_node);
    break;

  default:
    fprintf(stderr, "[TREE MD] Unknown hash: %u\n",
            hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

/* Loop over all children of the TSNode and convert them. */
static void _children(Node *parent, const char *source, TSNode ts_parent) {
  Node *child;
  TSNode ts_child;

  for (int i = 0; i < ts_node_named_child_count(ts_parent); i++) {
    ts_child = ts_node_named_child(ts_parent, i);
    child = _node(source, ts_child);
    add_child(parent, child);
  }
}

/*
 * *Main*
 */

Node *convert_tree_md(const char *source, TSTree *tree) {
  Node *root;
  TSNode ts_root;
  unsigned int hash_document;

  ts_root = ts_tree_root_node(tree);
  hash_document = hash(ts_node_type(ts_root));

  assert(hash_document == HASH_DOCUMENT);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = create_node(hash_document, NULL);
  _children(root, source, ts_root);
  return root;
}

// void _convert_emphasis(FILE *file, const char *source, TSNode node) {
//   assert(ts_node_named_child_count(node) == 2);
//
//   TSNode node_1 = ts_node_named_child(node, 0);
//   TSNode node_2 = ts_node_named_child(node, 1);
//   char *emph_1 = node_text(source, node_1);
//   char *emph_2 = node_text(source, node_2);
//   char *text = extract_text(source, ts_node_end_byte(node_1),
//                             ts_node_start_byte(node_2));
//   assert(strcmp(emph_1, emph_2) == 0);
//
//   if (strcmp(emph_1, "*") == 0) {
//     fprintf(file, "<strong>%s</strong>", text);
//   } else if (strcmp(emph_2, "_") == 0) {
//     fprintf(file, "<em>%s</em>", text);
//   } else
//     assert(false);
//
//   free(text);
//   free(emph_2);
//   free(emph_1);
// }
//
// Node *_convert_inline(const char *source, TSNode node) {
//   char *text;
//   TSNode child;
//   unsigned int start = ts_node_start_byte(node);
//   unsigned int end;
//
//   for (int i = 0; i < ts_node_named_child_count(node); i++) {
//     child = ts_node_named_child(node, i);
//
//     // Copy the content from the current start to the next child.
//     end = ts_node_start_byte(child);
//     text = extract_text(source, start, end);
//     fprintf(file, "%s", text);
//     free(text);
//
//     // Convert the child.
//     switch (hash(ts_node_type(child))) {
//     case HASH_EMPHASIS:
//       _convert_emphasis(file, source, child);
//       break;
//
//     default:
//       fprintf(stderr, "%s", ts_node_type(child));
//       assert(false);
//     }
//
//     // Update starting point for next iteration.
//     start = ts_node_end_byte(child);
//   }
//
//   // Copy the rest of the inline node.
//   end = ts_node_end_byte(node);
//   text = extract_text(source, start, end);
//   fprintf(file, "%s", text);
//   free(text);
// }
//
// void _convert_named_children(FILE *file, const char *source, TSNode node) {
//   for (int i = 0; i < ts_node_named_child_count(node); i++)
//     convert_tree(file, source, ts_node_named_child(node, i));
// }
