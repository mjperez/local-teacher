# scripts/branch-inventory.ps1
param (
    [string]$MainBranch = "main"
)

$results = [System.Collections.Generic.List[PSCustomObject]]::new()

$branches = git for-each-ref --format='%(refname:short)' refs/heads/ | 
    Where-Object { $_ -ne $MainBranch -and $_ -notmatch '^backup/' }

foreach ($branch in $branches) {
    $ahead = [int](git rev-list --count "$MainBranch..$branch").Trim()
    $behind = [int](git rev-list --count "$branch..$MainBranch").Trim()
    $lastDate = (git log -1 --format="%cs" $branch).Trim()
    $lastMsg = (git log -1 --format="%s" $branch).Trim()

    $status = if ($ahead -eq 0) {
        "MERGED"
    } elseif ($behind -eq 0) {
        "CLEAN_AHEAD"
    } else {
        "DIVERGED"
    }

    $results.Add([PSCustomObject]@{
        Branch         = $branch
        Ahead          = $ahead
        Behind         = $behind
        LastCommitDate = $lastDate
        Status         = $status
        LastSubject    = $lastMsg
    })
}

$results | Sort-Object LastCommitDate -Descending | Format-Table -AutoSize