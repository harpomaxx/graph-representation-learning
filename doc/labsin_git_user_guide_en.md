# LABSIN GIT workflow
At LABSIN we use a work methodology that relies heavily on [github](https://github.com) (or eventually [gitlab](https://gitlab.com)). [Git](https://git-scm.com/) is a version control system that allows you to keep track of the changes made in the source code.

Unlike CVS, SVN or TFS (scm from microsoft), `git` is a distributed version control system: that is, there is a repo called `origin`, which is like the point where eventually the work of all, but then each has a local repo (at least). That is why the changes are first `committed` (to your local repo), and then a `push` is made to send those changes.

## Github
Github is a website that relies on git and allows you to keep track of changes made to the code. Its main advantage is that it allows you to work collaboratively. In addition, github relies heavily on a ticket system or *ssues* where some characteristic or feature that must be fulfilled is indicated. These issues are assigned to the different participants, and their status changes as the resolution progresses. Being `Closed` the final status of the issue when the task has already been concluded. Most of these issues will be related to code implementation, ** BUUUUT**, it is not the only use. We actually use issues for everything. For example, to indicate that some task of reading / writing a paper related to the project is missing or make a presentation, prepare a lecture about the project topics, etc.

### The workflow
The idea is to follow the [github worflow](http://https://docs.github.com/en/get-started/quickstart/github-flow "worflow").

There is nothing new about our methodloogy, but since the github doc is large, here it is a summary about it.

1. We work hierarchically with 2 branches (in parallel)`
`master`: Production
`develop`: Integration branch used in all the projects

2. For each new feature we need to create a new topic branch These branches should always be created from `develop`. The process is the following:
a) `git checkout develop`  (Change to branch develop)
b)  `git pull origin develop`  (Get the last from origin  repo)
c)  `git checkout -b feature/<nombre-descriptivo-de-mi-feature> develop` (create a new branch from develop and change to that new branch) 
d) Then you work on the new branch
d) `git commit -am ".... <comentario descriptivo de lo que hiciste...>"`  (commit changes in your local repo) In the comment you can make a link to the related issue using #
e)`git push origin feature/<nombre....> ` (send all branch content to remote repo `origin`. creating the branch if it doesn't exist or updating it if it exists already)
f) Then the PR (Pull Request) is created from the github site and the comments of the others are expected.
g) When the PR is approved, you can do the merge

Despues, si alguien crea un branch y lo pushea a `origin`, y vos queres bajartelo a tu repo local, haces primero `git fetch origin`,  y después podés hacer `git checkout <branch>`  o alternativamente haces `git checkout --track origin/<branch>`

Later, if someone creates a new branch and sets it to `origin`, and you want to download it to your local repo, you first do` git fetch origin`, and then you can do `git checkout <branch>` or alternatively do `git checkout - -track origin / <branch>

 ### About  releases
When you are going to make a release, you typically do a single PR from `develop` to `master`, check and merge in `master`. 

Then to keep the version history you usually add tags in `master` (you can add them in any branch anyway if you need them). The tags are like markers in the history and then if necessary you can create branches off of those tags.

### Useful commands
#### Delete N commits
`git reset --hard HEAD~2` command moves the current branch backward by two commits.
`git push origin -f` 
