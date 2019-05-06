eval $(ssh-agent -s)

eval $(ssh-add /c/Users/paule/.ssh/id_rsa)

git push $1 $2
