# RMIT COSC1125/1127 AI'22 - Project 1 - Search

You must read fully and carefully the assignment specification and instructions detailed in this file. You are NOT to modify this file in any way, except if instructed by the teaching staff in writing.

* **Course:** [COSC1127/1125 Artificial Intelligence](http://www1.rmit.edu.au/courses/004123) @ Semester 2, 2022
* **Instructor:** Prof. Sebastian Sardina
* **Deadline:** Sunday August 7th, 2022 @ 11:59pm (end of Week 3)
* **Course Weight:** 7%
* **Assignment type:**: Individual
* **CLOs covered:** 2, 3, 4 and 5
* **Submission method:** via git tagging (see below for instructions)

The **aim of this project** is to get you acquainted with AI search techniques and how to derive heuristics in Pacman, as well as to understand the Python-based Pacman infrastructure.

 <p align="center">
    <img src="logo-p1.jpg" alt="logo project 1">
 </p>


## Your task

In this assessment, **your task** is to fully complete Q1-Q8 in the original [UC Pacman Project 1 - Search](http://ai.berkeley.edu/search.html) plus the additional Q9 described below.

* You **must build and submit your solution** using the sample code we provide you in this repository, which is different from the original UCB code base. 

* You should **only work and modify** files `search.py` and `searchAgents.py` in doing your solution. Do not change any of the other Python files in this distribution.

* You **must follow the course book (AIMA)** the search algorithms. Note the the tests will be checking whether you have implemented a _particular_ search algorithm  accurately, not just that you are solving the search problem. Implementing a different variant of the algorithm can result in less or none points awarded.

* Your code **must run _error-free_ on Python 3.6/3.8**. Staff will not debug/fix any code. Using a different version will risk your program not running with the Pacman infrastructure or autograder and may risk losing (all) marks. You can install Python 3.6/3.8 from the [official site](https://www.python.org/dev/peps/pep-0494/), or set up a [Conda environment](https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c/) or an environment with [PIP+virtualenv](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/). See also [these questions](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-PACMAN.md#what-version-of-python-should-i-use) in the FAQ for more info.

* You should **never tamper with the Pacman infrastructure**, neither at the source code level (e.g., changing files other than the ones for the task) nor at the run-time level (e.g., changing infrastructure properties or catching all exceptions with bare `except:` code). Check [this](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-PACMAN.md#can-i-change-the-pacman-infrastructure-at-run-time) and [this](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-PACMAN.md#can-i-use-catch-all-exceptions-in-my-code-or-exceptions-from-the-infrastructure) questions in the FAQ on this and ask if in doubt.

* Your code **must not contain any personal information**, like your student number or your name. That info should go in the [STUDENT.md](STUDENT.md) file, as per instructions below. If you use an IDE that inserts your name, student number, or username, you should disable that.

* You **must follow good SE practice** during you development; please refer to Marking criteria below.

* You are free to **add additional testing scenarios** under the `test_case/` folder.

### Question 9 (5 marks)

Implement the **Iterative Deepening** tree search algorithm as per the AIMA book, by inserting your code into function `iterativeDeepeningSearch` within file `search.py`.

You should be able to do preliminary testing of the algorithm using the following command:

```shell
$ python pacman.py -l tinyMaze -p SearchAgent -a fn=ids
```

Other layouts are available in the layouts directory and some preliminary testing cases in the feedback autograder, but you can easily create you own layouts and test cases!

## Marking criteria

We will follow the marking weights specified in the official project instructions for Q1-Q8, plus the additional Q9 described above. Observe that while the autograder is a useful indication of your performance, it is only a feedback tools and _may not represent the ultimate mark_. The **ultimate mark** will be provided to you after marking. We reserve the right to run more tests, inspect your code and repo manually, and arrange for a face-to-face meeting for a discussion and demo of your solution if needed. **We will also run [Codequiry](https://codequiry.com/)** on all submitted solutions (see _Academic Dishonesty_ below).

You must also **follow good SE practices**, including good use of git version control during your development such as:

* _Commit early, commit often, commit meaningful:_ single or few commits with all the solution or big chunks of it, is not good practice.
* _Use meaningful commit messages:_ as a comment in your code, the message should clearly summarize what the commit is about. Messages like "fix", "work", "commit", "changes" are poor and do not help us understand what was done.
* _Use atomic commits:_ avoid commits doing many things; let alone one commit solving many questions of the project. Each commit should be about one (little but interesting) thing.

**We will inspect the commit history** (and **GitHub team** if a team project) to check for high-quality SE practices and evidence of _meaningful contributions of all members_. The results of this check can affect the overall mark of the project and point deductions may be applied when poor SE practices have been used or no evidence of contributions can be found. For example, few commits with a lot of code changes, or no or poor communication in the corresponding GitHub team may result in deductions, even if the performance is excellent.

## Submission Instructions

To **submit your assignment** you must complete the following four steps:

1. Fully complete the [STUDENT.md](STUDENT.md) file with the team details of the submission (even if it is an individual assignment).
2. Check that your solution runs on Python 3.6/3.8 and that your source code does not include personal information, like your student number or name.
3. Tag the commit version in the `master` branch you want to be submit with tag name `submission` (case sensitive).
    * The commit and tagging should be dated _before_ the deadline.
    * Make sure your submission is merged into the `master` branch, which should contain your latest stable version.
    * Note that a _tag_ is a name given to a specific commit in your git history. It is  NOT a branch nor a commit message nor a release. If you create a branch, release, or commit message with the text "`submission`", that will not be counted as tags and no marking will be done.
    * For more info on (re)tagging, please read the [these two entries](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-PACMAN.md#how-do-i-submit-my-project-solution-in-my-git-repository) in the Pacman FAQ.
4. Fill the [Project 1 Certification Form](https://forms.gle/1jTh43aiwSvUcY9b6).
    * Non-certified submissions will not be marked and will attract zero marks.

**IMPORTANT:** Submissions not compatible with the instructions above will attract zero marks and do not warrant a re-submission. Staff will not debug or fix your submission. Read carefully and ask for help (in forum or drop-in lab) if needed.

## Important information

**About this repo:** You must ALWAYS keep your fork **private** and **never share it** with anybody in or outside the course, except your teammates, _even after the course is completed_. You are not allowed to make another repository copy outside the provided GitHub Classroom without the written permission of the teaching staff. Please respect the [authors request](http://ai.berkeley.edu/project_instructions.html):

> **_Please do not distribute or post solutions to any of the projects._**

**Corrections:** From time to time, students or staff find errors (e.g., typos, unclear instructions, etc.) in the assignment specification. In that case, a corrected version of this file will be produced, announced, and distributed for you to commit and push into your repository.  Because of that, you are NOT to modify this file in any way to avoid conflicts.

**Late submissions & extensions:** A penalty of 10% of the maximum mark per day will apply to late assignments up to a maximum of five days, and 100% penalty thereafter (see [this question](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-COURSE.md#can-i-submit-late-what-is-the-penalty) in the course FAQs. Extensions will only be permitted in _exceptional_ circumstances; see [this question](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-COURSE.md#what-is-the-policy-on-special-consideration) in the course FAQs.

**Academic Dishonesty:** This is an advanced course, so we expect full professionalism and ethical conduct.  Plagiarism is a serious offense. Please **don't let us down and risk our trust**. Sophisticated _plagiarism detection_ software via [Codequiry](https://codequiry.com/) will be used in this edition to check submitted code against other submissions in the class as well as resources available on the web. These systems are really smart, so just do not risk it and keep professional and safe. We trust you all to submit your own work only; again, don't let us down. If you do, we will pursue the strongest consequences available to us according to the **University Academic Integrity policy**. In a nutshell, **never look at solution done by others**, either in (e.g., classmate) or outside (e.g., web) the course: they have already done their learning, this is your opportunity! If you refrain from this behavior, you are safe. For more information on this see file [Academic Integrity](ACADEMIC_INTEGRITY.md).

**We are here to help!:** We are here to help you! But we don't know you need help unless you tell us. We expect reasonable effort from your side, but if you get stuck or have doubts, please seek help. We will run a drop-in lab to support these projects, so use that! While you have to be careful to not post spoilers in the forum, you can always ask general questions about the techniques that are required to solve the projects. If in doubt whether a questions is appropriate, post a Private post to the instructors. There is also a dedicated [**PacMan FAQ**](https://github.com/RMIT-COSC1127-1125-AI22/AI22-DOC/blob/main/FAQ-PACMAN.md) available to record common questions, check them before asking, your question may already be there!


**Silent Policy:** A silent policy will take effect **48 hours** before this assignment is due. This means that no question about this assignment will be answered, whether it is asked on the newsgroup, by email, or in person.

## AI'22 Code of Honour

We expect every RMIT student taking this course to adhere to the **Code of Honour** under which every learner-student should:

* Submit their own original work.
* Do not share answers with others.
* Report suspected violations.
* Not engage in any other activities that will dishonestly improve their results or dishonestly improve or damage the results of others.

Unethical behaviour is extremely serious and consequences are painful for everyone. We expect enrolled students/learners to take full **ownership** of your work and **respect** the work of teachers and other students.

**I hope you enjoy the project and learn from it**, and if you still **have doubts about the project and/or this specification** do not hesitate asking in the [EdStem Course Discussion Forum](https://edstem.org/au/courses/8118) and we will try to address it as quickly as we can!

**GOOD LUCK!**

## Acknowledgements

This is [UC Pacman Project 1 - Search](http://ai.berkeley.edu/search.html) from the set of [UC Pacman Projects](http://ai.berkeley.edu/project_overview.html).  We are very grateful to UC Berkeley CS188 for developing and sharing their system with us for teaching and learning purposes.
