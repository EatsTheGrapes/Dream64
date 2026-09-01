# assoc_positional_key_assign

Ground truth for `L[n] = x` where `n` is a positive integer within `length(L)`
and iteration position `n` currently holds an associative pair `K = V`.

Observed on BYOND 516.1680 (see `expected-byond-516.1680.txt`):

- The slot keeps its position and the list keeps its length.
- The key at position `n` becomes `x`; the association is dropped, so the entry
  now has a null value until something writes through the new key.
- No key deduplication: assigning a key that already exists elsewhere leaves two
  entries with that key. Assigning the same key back still clears the value.
- A numeric `x` is accepted as the new key; reading `L[n_bigger_than_len]`
  afterwards is still an out-of-bounds runtime error (it is a key, not a
  positional element).

Run:

```powershell
Push-Location fixtures\oracle\assoc_positional_key_assign
& 'C:\Program Files (x86)\BYOND\bin\dm.exe' assoc_positional_key_assign.dme
& 'C:\Program Files (x86)\BYOND\bin\DreamDaemon.exe' assoc_positional_key_assign.dmb -trusted -close
Get-Content assoc_positional_key_assign.out
Pop-Location
```
