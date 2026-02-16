test -f /etc/profile.dos && . /etc/profile.dos

# Some applications read the EDITOR variable to determine your favourite text
# editor. So uncomment the line below and enter the editor of your choice :-)
#export EDITOR=/usr/bin/vim
#export EDITOR=/usr/bin/mcedit

# add aliases if there is a .aliases file
test -s ~/.alias && . ~/.alias
