fetch("http://localhost:8000/api/validate")
 .then(r=>r.json())
 .then(d=>{ document.querySelector("#metrics .card-body").innerHTML =
   `<b>Result:</b> ${d.result}<br><b>RMSE:</b> ${d.rmse}<br><b>MAE:</b> ${d.mae}`; })
 .catch(e=>console.error(e));
