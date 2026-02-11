$grid = @"

---

<div class="mt-12 border-t border-gray-200 dark:border-gray-700 pt-8">
    <h3 class="text-xl font-bold text-slate-800 dark:text-slate-200 mb-6 text-center uppercase tracking-widest">Quick Switch: Explore Other Tracks</h3>
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <a href="Week_12_Project_IDS.html" class="flex items-center p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-secondary transition-all group no-underline">
            <div class="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center mr-3 group-hover:bg-secondary/20 transition-colors">
                 <svg class="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.13-2.054-.382-3.016z"></path></svg>
            </div>
            <span class="text-sm font-bold text-slate-700 dark:text-slate-300 group-hover:text-secondary transition-colors line-clamp-2">Security (IDS)</span>
        </a>
        <a href="Week_12_Project_CNN.html" class="flex items-center p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-secondary transition-all group no-underline">
            <div class="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center mr-3 group-hover:bg-secondary/20 transition-colors">
                 <svg class="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
            </div>
            <span class="text-sm font-bold text-slate-700 dark:text-slate-300 group-hover:text-secondary transition-colors line-clamp-2">Computer Vision</span>
        </a>
        <a href="Week_12_Project_NLP.html" class="flex items-center p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-secondary transition-all group no-underline">
            <div class="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center mr-3 group-hover:bg-secondary/20 transition-colors">
                 <svg class="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
            </div>
            <span class="text-sm font-bold text-slate-700 dark:text-slate-300 group-hover:text-secondary transition-colors line-clamp-2">NLP Sentiment</span>
        </a>
        <a href="Week_12_Project_Recommender.html" class="flex items-center p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-secondary transition-all group no-underline">
            <div class="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center mr-3 group-hover:bg-secondary/20 transition-colors">
                 <svg class="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.382-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"></path></svg>
            </div>
            <span class="text-sm font-bold text-slate-700 dark:text-slate-300 group-hover:text-secondary transition-colors line-clamp-2">Recommender</span>
        </a>
        <a href="Week_12_Project_Forecasting.html" class="flex items-center p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-secondary transition-all group no-underline">
            <div class="w-10 h-10 rounded-full bg-secondary/10 flex items-center justify-center mr-3 group-hover:bg-secondary/20 transition-colors">
                 <svg class="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
            </div>
            <span class="text-sm font-bold text-slate-700 dark:text-slate-300 group-hover:text-secondary transition-colors line-clamp-2">Forecasting</span>
        </a>
    </div>
</div>
"@

$files = @(
    "Week 12 - Final Project & Presentation\Week_12_Project_Recommender.md",
    "Week 12 - Final Project & Presentation\Week_12_Project_Forecasting.md"
)

foreach ($file in $files) {
    Add-Content -Path $file -Value $grid -NoNewline
}
