import setuptools




with open("README.md","r",encoding="utf-8") as f:
    long_discription = f.read()

__version__ = "0.0.0"

Repo_Name = "Shasmeet-Shinde"
Author_User_Name = "MotoMLTaskphase"
SRC_Repo = "mlproject"
Author_Email="shasmeet.shasmeet@gmail.com"

setuptools.setup(
    name=SRC_Repo,
    version=__version__,
    author=Author_User_Name,
    author_email=Author_Email,
    description="a small py pacakge for ml app",
    long_description=long_discription,
    long_description_content="text/markdown",
    url=f"https://github.com/{Author_User_Name}/{Repo_Name}",
    project_urls={"Bug Tracker": f"https://github.com/{Author_User_Name}/{Repo_Name}/issues",},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")

)