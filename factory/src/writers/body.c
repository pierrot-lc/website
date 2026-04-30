#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "hash.h"
#include "parsers/bibtex.h"
#include "parsers/markdown.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "writers/body.h"

static void bibliography_cite(FILE *, Node *);

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
    case HASH_LINK_BIBLIOGRAPHY:
      bibliography_cite(file, child);
      return;

    case HASH_LINK_DESTINATION:
      destination = child;
      break;

    case HASH_LINK_LABEL:
      destination =
          search_label_destination(tree_root(node), child->data.content);
      break;

    case HASH_LINK_TEXT:
      text = child;
      break;

    default:
      fprintf(stderr, "[WRITER LINK] Unexpected hash type: %u\n", child->code);
      assert(false);
    }
  }

  if (destination != NULL)
    fprintf(file, "<a href=\"%s\">", destination->data.content);
  else
    fprintf(file, "<a>");

  if (text != NULL)
    write_children(file, text);

  fprintf(file, "</a>");
}

static void code_block(FILE *file, Node *node) {
  if (node->data.content != NULL)
    fprintf(file, "<pre><code class=\"language-%s\">", node->data.content);
  else
    fprintf(file, "<pre><code>");
  fprintf(file, "%s", node->children[0]->data.content);
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
      link = search_label_destination(tree_root(node), child->data.content);
      break;
    }
  }

  assert(link != NULL);

  if (desc != NULL)
    fprintf(file, "<img src=\"%s\" alt=\"%s\">", link->data.content,
            desc->data.content);
  else
    fprintf(file, "<img src=\"%s\">", link->data.content);
}

static void latex(FILE *file, Node *node) {
  if (node->code == HASH_LATEX_DISPLAY)
    fprintf(file, "<span class=\"latex-display\">%s</span>",
            node->data.content);
  else if (node->code == HASH_LATEX_INLINE)
    fprintf(file, "<span class=\"latex-inline\">%s</span>", node->data.content);
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

static void table(FILE *file, Node *node) {
  assert(node->code == HASH_PIPE_TABLE);
  Table *table = node->data.table;

  fprintf(file, "<table>\n");
  fprintf(file, "<thead>\n");
  fprintf(file, "<tr>\n");
  for (int j = 0; j < table->ncols; j++) {
    fprintf(file, "<th scope=\"col\" class=\"align-%s\">",
            table->columns[j]->alignment);
    write_children(file, table->columns[j]->cell);
    fprintf(file, "</th>\n");
  }
  fprintf(file, "</tr>\n");
  fprintf(file, "</thead>\n");

  fprintf(file, "<tbody>\n");
  for (int i = 0; i < table->nrows; i++) {
    fprintf(file, "<tr>\n");
    for (int j = 0; j < table->ncols; j++) {
      fprintf(file, "<td class=\"align-%s\">", table->columns[j]->alignment);
      write_children(file, table->cells[i][j]);
      fprintf(file, "</td>\n");
    }
    fprintf(file, "</tr>\n");
  }
  fprintf(file, "</tbody>\n");
  fprintf(file, "</table>\n");
}

void write_tree(FILE *file, Node *node) {
  switch (node->code) {
  case HASH_ATX_HEADING:
    balise(file, node, node->data.content);
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

  case HASH_PIPE_TABLE:
    table(file, node);
    break;

  case HASH_PIPE_TABLE_ROW:
    fprintf(file, "<tr>\n");
    write_children(file, node);
    fprintf(file, "</tr>\n");
    break;

  case HASH_STRONG_EMPHASIS:
    balise(file, node, "strong");
    break;

  case HASH_THEMATIC_BREAK:
    fprintf(file, "<hr />\n");
    break;

  case HASH_HTML_BLOCK:
  case HASH_TEXT:
    fprintf(file, "%s", node->data.content);
    break;

  case HASH_DOCUMENT:
  case HASH_INLINE:
  case HASH_SECTION:
    write_children(file, node);
    break;

  case HASH_BIBTEX_DOCUMENT:
  case HASH_BLOCK_CONTINUATION:
  case HASH_BLOCK_QUOTE_MARKER:
  case HASH_LINK_REFERENCE_DEFINITION:
  case HASH_MINUS_METADATA:
  case HASH_PIPE_TABLE_DELIMITER_ROW:
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
        fprintf(file, "<li>%s</li>\n",
                tags->children[i]->children[0]->data.content);
      }

    fprintf(file, "</ul>\n");
  }
}

static void bibliography_cite(FILE *file, Node *node) {
  Author *authors;
  Node *entry, *year;

  entry = search_bibliography_entry(tree_root(node), node->data.content);
  if (entry == NULL) {
    fprintf(stderr, "Missing a bibliography entry: %s\n", node->data.content);
    assert(false);
  }
  if ((year = get_field(entry, "year")) == NULL) {
    fprintf(stderr, "'year' not found for bibtex entry %s\n",
            entry->children[1]->data.content);
    assert(false);
  }
  authors = parse_authors(entry);

  fprintf(file, "<a href=\"#%s\">", node->data.content);
  fprintf(file, "<cite>%s", authors->lastname);
  if (authors->next != NULL)
    fprintf(file, " et al.");
  fprintf(file, ", %s</cite>", year->children[1]->data.content);
  fprintf(file, "</a>");

  mark_cited(entry); // To appear at the end of the bibliography.
  free_authors(authors);
}

void bibliography_entry(FILE *file, Node *entry) {
  Author *authors;
  Node *key, *title, *year, *aux;

  if ((key = search_node(entry, HASH_KEY_BRACE)) == NULL)
    assert(false);
  if ((title = get_field(entry, "title")) == NULL)
    assert(false);
  if ((year = get_field(entry, "year")) == NULL)
    assert(false);

  authors = parse_authors(entry);

  fprintf(file, "<li id=\"%s\">\n", key->data.content);
  fprintf(file, "<span class=\"title\">%s</span>\n",
          title->children[1]->data.content);
  fprintf(file, "<br>\n");
  fprintf(file, "<span class=\"aux\">");
  while (authors != NULL) {
    fprintf(file, "%s, %c., ", authors->lastname, authors->firstname[0]);
    authors = authors->next;
  }
  fprintf(file, "%s", year->children[1]->data.content);
  if ((aux = get_field(entry, "booktitle")) != NULL)
    fprintf(file, ", %s", aux->children[1]->data.content);
  if ((aux = get_field(entry, "publisher")) != NULL)
    fprintf(file, ", %s", aux->children[1]->data.content);
  if ((aux = get_field(entry, "doi")) != NULL)
    fprintf(file, ", DOI: %s", aux->children[1]->data.content);
  fprintf(file, "</span>\n");
  fprintf(file, "</li>\n");

  free_authors(authors);
}

void write_bibliography(FILE *file, Node *tree) {
  Node *entries, *entry, *marked;

  if ((entries = search_node(tree, HASH_BIBTEX_DOCUMENT)) == NULL)
    return;

  fprintf(file, "\n<section id=\"bibliography\">\n");
  fprintf(file, "<h2>References</h2>\n");
  fprintf(file, "<ol>\n");
  for (int i = 0; i < entries->child_count; i++) {
    entry = entries->children[i];
    assert(entry->code == HASH_ENTRY);

    if ((marked = get_field(entry, "cited?")) != NULL &&
        strcmp(marked->children[1]->data.content, "yes") == 0)
      bibliography_entry(file, entry);
  }
  fprintf(file, "</ol>\n");
  fprintf(file, "</section>\n");
}
