eval $(ssh-agent -s)

eval $(ssh-add /c/Users/paule/.ssh/gradieml_id_rsa)

git pull
