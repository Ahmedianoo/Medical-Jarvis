# JARVIS-OPTIMIZERS


**please follow this git flow**
1) pull latest main from origin 
2) checkout main
3) create a new branch using [ git checkout -b branchname ]
branch name should follow this convention 
feature/added-nav-bar
fix/create-users-endpoint
chore/removed-unnecessary-file 
4) work on your new branch , when you finish a portion of your work
use (git add .) to add your changed files to stage
git commit -m "commit message"  , commit message should follow the following convention feat(title) : message
5) while working on your branch , other team members could merge other PRs , this will make the tip of your branch outdated (the commit on main branch where you checked out your new branch)
6) you need then to fetch the latest updates from origin so you can track the latest changes using --> git fetch origin
7) you need then to make sure you are checked out on your feature branch , then apply -->  git rebase origin/main , this will update the base of your branch , as if you opened your branch from the latest commit on the main branch
once rebased , you can now push your branch if you are done with your feature

You can do this using vscode, you do not need to do it with command line. what is written is just for explanation.
