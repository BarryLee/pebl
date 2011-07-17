#!/bin/sh

NUM_PARAMS=2

self=`basename $0`

usage () {
  echo "Usage: $self -f <host>"
  echo "       $self -t <host>"
  echo
  echo "Options:"
  echo "  -f HOST    Set time as the remote HOST"
  echo "  -t HOST    Set the remote HOST's time as localhost"
}

synctime () {
  if [ $1 = 'f' ]
  then
    date -s "$(ssh $2 "date +\"%Y%m%d %H:%M:%S\"")"
  elif [ $1 = 't' ]
  then
    t=$(date +"%Y%m%d %H:%M:%S")
    ssh $2 "date -s \"$t\""
  fi
}

#if [ $# -ne $NUM_PARAMS ]
#then
#  usage
#  exit 1
#fi  

while getopts "f:t:h" arg
do
    case $arg in
         f)
            echo "Set time to be the same as $OPTARG's"
            synctime "f" $OPTARG
            ;;
         t)
            echo "Set time of $2 to be the same as mine"
            synctime "t" $OPTARG
            ;;
         h)
            usage
            ;;
         ?)
            usage
            exit 1
            ;;
    esac
done


#date -s "$(ssh root@10.9.9.5 "date +\"%Y%m%d %H:%M:%S\"")"

