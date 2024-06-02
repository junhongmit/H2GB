$(document).ready(function() {
  // Remove "Build with Sphinx" text in the footer:
  var elem = $("footer div[role='contentinfo']").get(0);
  while (elem.nextSibling)
    elem.nextSibling.remove();

  // Code block header:
  var $highlight = $("div.rst-content div.highlight");
  $highlight.prepend("<div class='mac-header'><span class='dot red'></span><span class='dot yellow'></span><span class='dot green'></span><span class='copy fa fa-clipboard' title='Copy to clipboard'></span></div>");

  // Copy text in code blocks:
  $highlight.find('span.copy').click(function() {
    navigator.clipboard.writeText($(this).parent().parent().text().slice(0, -1));
  });

  // Set monospace fonts:
  addMonospaceFont("li a");
  addMonospaceFont("li.breadcrumb-item");
  addMonospaceFont("div.rst-content h1");
});

function addMonospaceFont(selector) {
  $(selector).each(function(i) {
    var text = $(this).text();
    if (text.startsWith("torch_geometric") || text.startsWith("pyg_lib") || text.startsWith("torch_frame")) {
      $(this).addClass("pyg-mono");

      // Delete main package names:
      text = $(this).contents()[0].textContent;
      var packages = text.split('.');
      if (packages.length > 2) {
        text = packages.slice(2).join('.')
        $(this).contents()[0].textContent = text;
      }
    }
    else if (text.startsWith("Source code")) {
      $(this).html(`${text.substr(0,16)}<span class="pyg-mono">${text.substr(16)}</span>`);
    }
  });
}
