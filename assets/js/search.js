document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('search-input');
  const resultsContainer = document.getElementById('results-container');
  
  let searchIndex = [];
  
  // Fetch search index
  fetch('/search.json')
    .then(response => response.json())
    .then(data => {
      searchIndex = data;
      
      // Initialize Lunr
      const idx = lunr(function() {
        this.ref('url');
        this.field('title');
        this.field('content');
        
        searchIndex.forEach(doc => {
          this.add(doc);
        }, this);
      });
      
      // Search handler
      searchInput.addEventListener('input', function() {
        const query = this.value.trim();
        resultsContainer.innerHTML = '';
        
        if (query.length < 2) return;
        
        const results = idx.search(query);
        
        results.forEach(result => {
          const doc = searchIndex.find(doc => doc.url === result.ref);
          
          const li = document.createElement('li');
          const a = document.createElement('a');
          a.href = doc.url;
          a.textContent = doc.title;
          li.appendChild(a);
          
          resultsContainer.appendChild(li);
        });
      });
    });
});
