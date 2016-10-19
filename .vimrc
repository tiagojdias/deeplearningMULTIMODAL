set encoding=utf-8
set termencoding=utf-8


" search properties
set incsearch
set ignorecase


" highlighting
set number

filetype plugin indent on
set foldmethod=indent
syntax on
set background=dark
" set cursorline
" color desert
" set cursorline
"  hi CursorLine ctermbg=0 ctermfg=15 term=bold cterm=bold guibg=red "8 = dark gray, 15 = white 
" hi CursorLine ctermbg=0 term=bold cterm=bold guibg=red 
" 8 = dark gray, 15 = white 
" hi Cursor ctermbg=90 ctermfg=8


" set cursorline
" hi CursorLine term=bold cterm=bold guibg=Grey40
"
"

set synmaxcol=0

" tabbing
set expandtab
set shiftwidth=2
set softtabstop=2

" folding settings
set foldmethod=indent
"set foldnestmax=10
"set nofoldenable
set foldlevel=99

" Enable folding with the spacebar
nnoremap <space> za

set wrap
set linebreak
set textwidth=0 
set wrapmargin=0
" set nowrap
set nolist  " list disables linebreak


" Speedup Vim!!!!!!
let loaded_matchparen=1 " Don't load matchit.vim (paren/bracket matching)
set noshowmatch         " Don't match parentheses/brackets
set nocursorline        " Don't paint cursor line
set nocursorcolumn      " Don't paint cursor column
set lazyredraw          " Wait to redraw
set scrolljump=8        " Scroll 8 lines at a time at bottom/top
let html_no_rendering=1 " Don't render italic, bold, links in HTML


" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" let Vundle manage Vundle, required
Plugin 'gmarik/Vundle.vim'
Plugin 'altercation/vim-colors-solarized'
Plugin 'airblade/vim-gitgutter'
let g:gitgutter_max_signs = 10000

Plugin 'MatlabFilesEdition'
Plugin 'vim-latex/vim-latex'

Plugin 'tmhedberg/SimpylFold'
let g:SimpylFold_docstring_preview = 1
let g:SimpylFold_fold_docstring = 0

Bundle 'Valloric/YouCompleteMe'
let g:ycm_autoclose_preview_window_after_completion=1
let g:ycm_min_num_of_chars_for_completion = 2

Plugin 'scrooloose/nerdtree'
Plugin 'jistr/vim-nerdtree-tabs'
let NERDTreeIgnore=['\.pyc$', '\~$'] "ignore files in NERDTree
map <silent> <C-n> :NERDTreeToggle<CR>


Plugin 'vim-misc'
Plugin 'xolox/vim-colorscheme-switcher'


" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required

" para LaTeX-Box
" if s:extfname ==? "tex"
"  let g:LatexBox_split_type="new"
" endif
let g:Tex_DefaultTargetFormat='pdf'
let g:tex_flavor='pdflatex -shell-escape'
let g:Tex_GotoError=0

if has("gui_running")

  colorscheme solarized
  let g:solarized_termcolors=256

  set guioptions-=m  "remove menu bar
  set guioptions-=T  "remove toolbar
  set guioptions-=r  "remove right-hand scroll bar
  set guioptions-=L  "remove left-hand scroll bar


  set background=dark

  if has("gui_gtk2")
    " set guifont=Ubuntu\ Mono\ for\ Powerline\ 14
    set guifont=Courier:14
    " https://github.com/pdf/ubuntu-mono-powerline-ttf -- to install folow
    " this link
    " set guifont=Courier:h14
  else
    set guifont=Courier:h14
  endif  
  " let g:Powerline_symbols = 'fancy'
  "   set encoding=utf-8
  "   set t_Co=256
  "   set fillchars+=stl:\ ,stlnc:\
  "   set term=xterm-256color
  "   set termencoding=utf-8
  " 
  "   "" POWER LINE CONFIG
  "   set laststatus=0
  "   let g:airline_powerline_fonts = 1
  "   let g:airline#extensions#tabline#enabled = 1
  "   "let g:airline#extensions#tabline#left_sep = ' '
  "   "let g:airline#extensions#tabline#left_alt_sep = '|'
  " 
  "   let g:airline_theme='luna'
  "   let g:airline#extensions#hunks#enabled = 1
  "   let g:airline#extensions#branch#enabled = 1
  "   let g:airline#extensions#branch#empty_message = 'out of GIT'
  "   "let g:airline_symbols.branch = ''
  "   "let g:airline_section_d = airline#section#create(['hunks', 'branch'])
  "   ""
  "   "let g:promptline_preset = 'clear' " or full
  "   "let g:promptline_theme = 'airline'

endif


" VIM-TEX CONFIG
" REQUIRED. This makes vim invoke Latex-Suite when you open a tex file.
filetype plugin on

" IMPORTANT: win32 users will need to have 'shellslash' set so that latex
" can be called correctly.
set shellslash

" IMPORTANT: grep will sometimes skip displaying the file name if you
" search in a singe file. This will confuse Latex-Suite. Set your grep
" program to always generate a file-name.
set grepprg=grep\ -nH\ $*

" OPTIONAL: This enables automatic indentation as you type.
filetype indent on

" OPTIONAL: Starting with Vim 7, the filetype of empty .tex files defaults to
" 'plaintex' instead of 'tex', which results in vim-latex not being loaded.
" The following changes the default filetype back to 'tex':
let g:tex_flavor='latex'

" Enable CursorLine
set cursorline
" Default Colors for CursorLine
" highlight  CursorLine ctermbg=Yellow ctermfg=None
" " Change Color when entering Insert Mode
" autocmd InsertEnter * highlight  CursorLine ctermbg=Green ctermfg=Red
" " Revert Color to default when leaving Insert Mode
" autocmd InsertLeave * highlight  CursorLine ctermbg=Yellow ctermfg=None
