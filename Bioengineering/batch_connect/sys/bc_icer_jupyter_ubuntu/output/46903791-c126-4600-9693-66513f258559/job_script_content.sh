
module purge\n
cd /mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559

# Export useful connection variables
export host
export port

# Generate a connection yaml file with given parameters
create_yml () {
  echo "Generating connection YAML file..."
  (
    umask 077
    echo -e "host: $host\nport: $port\npassword: $password" > "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/connection.yml"
  )
}

# Cleanliness is next to Godliness
clean_up () {
  echo "Cleaning up..."
  [[ -e "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/clean.sh" ]] && source "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/clean.sh"
  [[ ${SCRIPT_PID} ]] && pkill -P ${SCRIPT_PID} || :
  pkill -P $$
  exit ${1:-0}
}

# Source in all the helper functions
source_helpers () {
  # Generate random integer in range [$1..$2]
  random_number () {
    shuf -i ${1}-${2} -n 1
  }
  export -f random_number

  port_used_python() {
    python -c "import socket; socket.socket().connect(('$1',$2))" >/dev/null 2>&1
  }

  port_used_python3() {
    python3 -c "import socket; socket.socket().connect(('$1',$2))" >/dev/null 2>&1
  }

  port_used_nc(){
    nc -w 2 "$1" "$2" < /dev/null > /dev/null 2>&1
  }

  port_used_lsof(){
    lsof -i :"$2" >/dev/null 2>&1
  }

  port_used_bash(){
    local bash_supported=$(strings /bin/bash 2>/dev/null | grep tcp)
    if [ "$bash_supported" == "/dev/tcp/*/*" ]; then
      (: < /dev/tcp/$1/$2) >/dev/null 2>&1
    else
      return 127
    fi
  }

  # Check if port $1 is in use
  port_used () {
    local port="${1#*:}"
    local host=$((expr "${1}" : '\(.*\):' || echo "localhost") | awk 'END{print $NF}')
    local port_strategies=(port_used_nc port_used_lsof port_used_bash port_used_python port_used_python3)

    for strategy in ${port_strategies[@]};
    do
      $strategy $host $port
      status=$?
      if [[ "$status" == "0" ]] || [[ "$status" == "1" ]]; then
        return $status
      fi
    done

    return 127
  }
  export -f port_used

  # Find available port in range [$2..$3] for host $1
  # Default host: localhost
  # Default port range: [54000..54500]
  # returns error code (0: success, 1: failed)
  # On success, the chosen port is echoed on stdout.
  find_port () {
    local host="${1:-localhost}"
    local min_port=${2:-54000}
    local max_port=${3:-54500}
    local port_range=($(shuf -i ${min_port}-${max_port}))
    local retries=1 # number of retries over the port range if first attempt fails
    for ((attempt=0; attempt<=$retries; attempt++)); do
      for port in "${port_range[@]}"; do
        if port_used "${host}:${port}"; then
continue
        fi
        echo "${port}"
        return 0 # success
      done
    done

    echo "error: failed to find available port in range ${min_port}..${max_port}" >&2
    return 1 # failure
  }
  export -f find_port

  # Wait $2 seconds until port $1 is in use
  # Default: wait 30 seconds
  wait_until_port_used () {
    local port="${1}"
    local time="${2:-30}"
    for ((i=1; i<=time*2; i++)); do
      port_used "${port}"
      port_status=$?
      if [ "$port_status" == "0" ]; then
        return 0
      elif [ "$port_status" == "127" ]; then
         echo "commands to find port were either not found or inaccessible."
         echo "command options are lsof, nc, bash's /dev/tcp, or python (or python3) with socket lib."
         return 127
      fi
      sleep 0.5
    done
    return 1
  }
  export -f wait_until_port_used

  # Generate random alphanumeric password with $1 (default: 8) characters
  create_passwd () {
    tr -cd 'a-zA-Z0-9' < /dev/urandom 2> /dev/null | head -c${1:-8}
  }
  export -f create_passwd
}
export -f source_helpers

source_helpers

# Set host of current machine
host=$(hostname)

[[ -e "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/before.sh" ]] && source "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/before.sh"

echo "Script starting..."
"/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/script.sh" &
SCRIPT_PID=$!

[[ -e "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/after.sh" ]] && source "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/after.sh"

# Create the connection yaml file
create_yml

# Wait for script process to finish
wait ${SCRIPT_PID} || clean_up 1

# Exit cleanly
clean_up



